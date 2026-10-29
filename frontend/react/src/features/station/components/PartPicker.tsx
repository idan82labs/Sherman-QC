/**
 * Station part picker: a modal dialog over the station screen for scans the
 * hotfolder held back (UNMATCHED / AMBIGUOUS / UNPARSEABLE / REVISION_MISMATCH
 * / PART_NOT_READY). The operator sees why the scan was held, its file name
 * and stats, the suggested parts first, can search every released part, picks
 * the recipe version and the scan stage, confirms, and the job is queued; the
 * live feed then shows it like any other job. A content problem (truncated,
 * unsupported, bad sidecar) can only be dismissed, with a reason.
 *
 * Assign / dismiss need an operator (`line.operate`): the signed-in user's
 * token when it has the role, else an operator signs in here; that token stays
 * in this component's memory and is dropped when the dialog closes.
 *
 * Accessibility: `role="dialog"` + `aria-modal`, labelled by its title; focus
 * moves into the dialog on open, Tab stays inside, Escape closes, and focus
 * returns to the button that opened it. The part list is a radiogroup.
 * Direction follows the screen (`dir` on the station root); part numbers,
 * file names and numbers are isolated left-to-right.
 */
import { useCallback, useEffect, useId, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertOctagon, CheckCircle2, Loader2, Search, X } from 'lucide-react'

import { assignHeld, dismissHeld, operatorLogin, searchReleasedParts, StationApiError } from '../api/stationApi'
import { dismissReasonText, issueReasonText, pickerErrorText, suggestionSourceText, t } from '../copy'
import {
  assignBody,
  canDismiss,
  defaultStage,
  DISMISS_REASONS,
  formatBytes,
  formatExtents,
  pickerRows,
  stageChoiceComplete,
  type DismissReason,
  type HeldScan,
  type InspectionStage,
  type PickRow,
} from '../model/partPicker'
import type { Lang } from '../model/types'

export interface PartPickerProps {
  lang: Lang
  stationId: string
  items: HeldScan[]
  /** The station's own token may assign (a signed-in operator / admin). */
  canOperate: boolean
  initialScanId?: string | null
  onClose: () => void
  /** Called after an assign or dismiss so the held list refreshes. */
  onResolved: () => void
}

type Step = 'pick' | 'confirm' | 'dismiss' | 'done'

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(id)
  }, [value, ms])
  return v
}

function timeOf(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
}

export function PartPicker({ lang, stationId, items, canOperate, initialScanId, onClose, onResolved }: PartPickerProps) {
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const openerRef = useRef<Element | null>(null)

  const [scanId, setScanId] = useState<string | null>(initialScanId ?? items[0]?.scan_file_id ?? null)
  const [step, setStep] = useState<Step>('pick')
  // The scan just resolved stays on screen with its "done" note even after the list refreshes without it.
  const [doneScan, setDoneScan] = useState<HeldScan | null>(null)
  const scan = items.find(i => i.scan_file_id === scanId) ?? (step === 'done' ? doneScan : null) ?? items[0] ?? null
  const [query, setQuery] = useState('')
  const debounced = useDebounced(query, 200)
  const [chosenSku, setChosenSku] = useState<string | null>(null)
  const [recipeId, setRecipeId] = useState<string | null>(null)
  const [stage, setStage] = useState<InspectionStage>('FINAL')
  const [stageIndex, setStageIndex] = useState<number | null>(null)
  const [dismissReason, setDismissReason] = useState<DismissReason | null>(null)
  const [dismissNote, setDismissNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState<'assigned' | 'dismissed' | null>(null)
  const [operator, setOperator] = useState<{ token: string; name: string } | null>(null)
  const [login, setLogin] = useState({ user: '', password: '' })
  const [loginError, setLoginError] = useState<string | null>(null)
  const mayOperate = canOperate || operator !== null

  const partsQuery = useQuery({
    queryKey: ['station-released-parts', stationId, debounced],
    queryFn: ({ signal }) => searchReleasedParts(stationId, debounced, signal),
    staleTime: 15_000,
    enabled: !!scan?.resolvable,
  })
  const rows: PickRow[] = useMemo(
    () => (scan ? pickerRows(scan.suggestions, partsQuery.data ?? [], query) : []),
    [scan, partsQuery.data, query],
  )
  const chosen = rows.find(r => r.part.sku === chosenSku) ?? null

  // Reset the choice when another held scan is selected.
  const resetFor = useCallback((id: string | null) => {
    setScanId(id)
    setStep('pick')
    setChosenSku(null)
    setRecipeId(null)
    setStageIndex(null)
    setDismissReason(null)
    setDismissNote('')
    setError(null)
    setDone(null)
  }, [])

  // Focus management: remember the opener, focus inside, restore on close.
  useEffect(() => {
    openerRef.current = document.activeElement
    const first = searchRef.current ?? dialogRef.current?.querySelector<HTMLElement>('button, input, select, textarea')
    first?.focus()
    return () => {
      const el = openerRef.current as HTMLElement | null
      if (el && typeof el.focus === 'function') el.focus()
    }
  }, [])

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape') {
      e.stopPropagation()
      onClose()
      return
    }
    if (e.key !== 'Tab' || !dialogRef.current) return
    const focusables = Array.from(
      dialogRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea, [tabindex="0"]'),
    )
    if (!focusables.length) return
    const first = focusables[0]
    const last = focusables[focusables.length - 1]
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault()
      first.focus()
    }
  }

  const choose = (row: PickRow) => {
    if (!row.part.assignable) return
    setChosenSku(row.part.sku)
    const active = row.part.versions.find(v => v.selectable) ?? null
    setRecipeId(active?.recipe_id ?? row.part.recipe_id)
    setStage(defaultStage(row.part))
    setStageIndex(null)
    setError(null)
  }

  const finish = (kind: 'assigned' | 'dismissed') => {
    setDoneScan(scan)
    setScanId(scan?.scan_file_id ?? null)
    setDone(kind)
    setStep('done')
    onResolved()
  }

  const fail = (err: unknown) => {
    const code = err instanceof StationApiError ? err.code : null
    if (err instanceof StationApiError && (err.status === 401 || err.status === 403)) {
      setOperator(null)
      setError(pickerErrorText('CAPABILITY_REQUIRED', lang))
    } else {
      setError(pickerErrorText(code, lang))
    }
  }

  const confirmAssign = async () => {
    if (!scan || !chosen || !recipeId || busy) return
    setBusy(true)
    setError(null)
    try {
      await assignHeld(stationId, scan.scan_file_id, assignBody(recipeId, chosen.part, stage, stageIndex), operator?.token ?? null)
      finish('assigned')
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  const confirmDismiss = async () => {
    if (!scan || !dismissReason || !canDismiss(dismissReason, dismissNote) || busy) return
    setBusy(true)
    setError(null)
    try {
      await dismissHeld(stationId, scan.scan_file_id, dismissReason, dismissNote, operator?.token ?? null)
      finish('dismissed')
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  const signIn = async (e: FormEvent) => {
    e.preventDefault()
    setLoginError(null)
    try {
      const res = await operatorLogin(login.user.trim(), login.password)
      if (res.role !== 'operator' && res.role !== 'admin') {
        setLoginError(t('pickerNotOperator', lang))
        return
      }
      setOperator({ token: res.token, name: res.name })
      setLogin({ user: '', password: '' })
    } catch {
      setLoginError(t('pickerSignInFailed', lang))
    }
  }

  // After a success: next held scan, or close.
  useEffect(() => {
    if (step !== 'done') return
    const id = window.setTimeout(() => {
      const next = items.find(i => i.scan_file_id !== scanId)
      if (next) resetFor(next.scan_file_id)
      else onClose()
    }, 1400)
    return () => window.clearTimeout(id)
  }, [step, items, scanId, resetFor, onClose])

  const operatorBlock = !mayOperate && (
    <form className="st-picker-auth" onSubmit={e => void signIn(e)} data-testid="picker-signin">
      <p>{t('pickerNeedOperator', lang)}</p>
      <label>
        <span>{t('pickerUser', lang)}</span>
        <input dir="ltr" autoComplete="username" value={login.user} onChange={e => setLogin(l => ({ ...l, user: e.target.value }))} />
      </label>
      <label>
        <span>{t('pickerPassword', lang)}</span>
        <input dir="ltr" type="password" autoComplete="current-password" value={login.password} onChange={e => setLogin(l => ({ ...l, password: e.target.value }))} />
      </label>
      <button type="submit" className="st-picker-btn" disabled={!login.user.trim() || !login.password}>{t('pickerSignIn', lang)}</button>
      {loginError && <p className="st-picker-error" role="alert">{loginError}</p>}
    </form>
  )

  const versionLabel = (v: { recipe_version: number; status: string }) =>
    `v${v.recipe_version} · ${v.status === 'active' ? t('pickerVersionActive', lang) : t('pickerVersionOld', lang)}`

  return (
    <div className="st-picker-backdrop" data-testid="part-picker-backdrop">
      <div
        ref={dialogRef}
        className="st-picker"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        data-testid="part-picker"
        data-step={step}
        onKeyDown={onKeyDown}
      >
        <header className="st-picker-head">
          <h2 id={titleId}>{t('pickerTitle', lang)}</h2>
          {operator && (
            <span className="st-picker-operator" data-testid="picker-operator">
              {t('pickerSignedInAs', lang)}: <bdi dir="ltr">{operator.name}</bdi>
            </span>
          )}
          <button type="button" className="st-picker-close" onClick={onClose} aria-label={t('pickerCancel', lang)} data-testid="picker-close">
            <X size={28} aria-hidden />
          </button>
        </header>

        <div className={`st-picker-body${items.length > 1 ? ' has-list' : ''}`}>
          {items.length > 1 && (
            <nav className="st-picker-list" aria-label={t('pickerHeldList', lang)} data-testid="picker-held-list">
              <h3>
                {t('pickerHeldList', lang)} <span className="st-num" dir="ltr">({items.length})</span>
              </h3>
              <ul>
                {items.map(i => (
                  <li key={i.scan_file_id}>
                    <button
                      type="button"
                      className={`st-picker-held${i.scan_file_id === scan?.scan_file_id ? ' is-current' : ''}`}
                      aria-current={i.scan_file_id === scan?.scan_file_id ? 'true' : undefined}
                      onClick={() => resetFor(i.scan_file_id)}
                      data-scan-id={i.scan_file_id}
                    >
                      <bdi dir="auto" className="st-picker-held-name">{i.file_name}</bdi>
                      <span className="st-picker-held-why">
                        {issueReasonText(i.reason_code, lang)} · <span className="st-num" dir="ltr">{timeOf(i.quarantined_at)}</span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </nav>
          )}

          {!scan ? (
            <p className="st-picker-empty">{t('pickerEmpty', lang)}</p>
          ) : (
            <>
              <section className="st-picker-scan" aria-label={t('pickerWhy', lang)} data-testid="picker-scan" data-reason={scan.reason_code}>
                <div className="st-picker-why">
                  <AlertOctagon size={26} aria-hidden />
                  <div>
                    <span className="st-picker-k">{t('pickerWhy', lang)}</span>
                    <strong data-testid="picker-why">{issueReasonText(scan.reason_code, lang)}</strong>
                    {scan.message && <span className="st-picker-msg" dir="ltr">{scan.message}</span>}
                  </div>
                </div>
                <dl className="st-picker-facts">
                  <div>
                    <dt>{t('pickerFile', lang)}</dt>
                    <dd dir="auto" className="st-picker-filename" data-testid="picker-filename">{scan.file_name}</dd>
                  </div>
                  <div>
                    <dt>{t('pickerReceived', lang)}</dt>
                    <dd className="st-num" dir="ltr">{timeOf(scan.received_at)}</dd>
                  </div>
                  {(scan.parsed.sku || scan.parsed.revision || (scan.parsed.candidates?.length ?? 0) > 0) && (
                    <div>
                      <dt>{t('pickerParsed', lang)}</dt>
                      <dd className="st-picker-chips">
                        {[scan.parsed.sku, scan.parsed.revision && `rev ${scan.parsed.revision}${scan.parsed.revision_status === 'MISMATCH' && scan.parsed.expected_revision ? ` (≠ ${scan.parsed.expected_revision})` : ''}`, scan.parsed.serial, scan.parsed.scan_label, ...(scan.parsed.candidates ?? [])]
                          .filter((x): x is string => !!x)
                          .map((x, k) => (
                            <bdi key={k} dir="ltr">{x}</bdi>
                          ))}
                      </dd>
                    </div>
                  )}
                  <div>
                    <dt>{t('pickerScanStats', lang)}</dt>
                    <dd className="st-picker-chips st-num" data-testid="picker-stats">
                      {scan.stats.format && <bdi dir="ltr">{scan.stats.format}</bdi>}
                      <bdi dir="ltr">{formatBytes(scan.stats.size_bytes)}</bdi>
                      {scan.stats.vertex_count != null && (
                        <span>
                          <bdi dir="ltr">{scan.stats.vertex_count.toLocaleString('en-US')}</bdi> {t('pickerPoints', lang)}
                        </span>
                      )}
                      {scan.stats.bbox_mm && <bdi dir="ltr">{formatExtents(scan.stats.bbox_mm)}</bdi>}
                    </dd>
                  </div>
                </dl>
              </section>

              <section className="st-picker-main">
                {step === 'pick' && scan.resolvable && (
                  <>
                    <label className="st-picker-search">
                      <span className="st-sr">{t('pickerSearchLabel', lang)}</span>
                      <Search size={24} aria-hidden />
                      <input
                        ref={searchRef}
                        type="search"
                        dir="ltr"
                        placeholder={t('pickerSearchLabel', lang)}
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                        data-testid="picker-search"
                        autoComplete="off"
                        spellCheck={false}
                      />
                      {partsQuery.isFetching && <Loader2 className="st-spin" size={22} aria-hidden />}
                    </label>
                    <div className="st-picker-rows" role="radiogroup" aria-label={t('pickerSearchLabel', lang)} data-testid="picker-rows">
                      {rows.length === 0 && !partsQuery.isFetching && <p className="st-picker-empty">{t('pickerNoParts', lang)}</p>}
                      {rows.map(row => {
                        const selected = row.part.sku === chosenSku
                        return (
                          <button
                            key={row.part.sku}
                            type="button"
                            role="radio"
                            aria-checked={selected}
                            aria-disabled={!row.part.assignable}
                            className={`st-picker-row${selected ? ' is-selected' : ''}${row.part.assignable ? '' : ' is-disabled'}`}
                            onClick={() => choose(row)}
                            data-sku={row.part.sku}
                            data-suggested={row.suggestion ? row.suggestion.source : undefined}
                          >
                            <bdi dir="ltr" className="st-picker-sku">{row.part.sku}</bdi>
                            <span className="st-picker-name" dir="auto">{row.part.part_name ?? ''}</span>
                            <span className="st-picker-meta" dir="ltr">
                              {row.part.revision ? `${t('pickerRevision', lang)} ${row.part.revision} · ` : ''}v{row.part.recipe_version}
                            </span>
                            {row.suggestion && (
                              <span className="st-picker-tag">
                                {t('pickerSuggested', lang)}: {suggestionSourceText(row.suggestion.source, lang)}
                              </span>
                            )}
                            {!row.part.assignable && <span className="st-picker-tag is-warn">{t('pickerNotReady', lang)}</span>}
                          </button>
                        )
                      })}
                    </div>
                    {chosen && (
                      <div className="st-picker-options" data-testid="picker-options">
                        <label>
                          <span>{t('pickerVersion', lang)}</span>
                          <select value={recipeId ?? ''} onChange={e => setRecipeId(e.target.value)} dir="ltr" data-testid="picker-version">
                            {chosen.part.versions.map(v => (
                              <option key={v.recipe_id} value={v.recipe_id} disabled={!v.selectable}>
                                {versionLabel(v)}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label>
                          <span>{t('pickerStage', lang)}</span>
                          <select
                            value={stage}
                            onChange={e => {
                              setStage(e.target.value as InspectionStage)
                              setStageIndex(null)
                            }}
                            data-testid="picker-stage"
                          >
                            {(['FINAL', 'IN_PROCESS'] as const).map(s => (
                              <option key={s} value={s}>
                                {s === 'FINAL' ? t('stageFinal', lang) : t('stageInProcess', lang)}
                                {s === defaultStage(chosen.part) ? ` (${t('pickerStageDefault', lang)})` : ''}
                              </option>
                            ))}
                          </select>
                        </label>
                        {stage === 'IN_PROCESS' && (
                          <label>
                            <span>{t('pickerBendsFormed', lang)}</span>
                            <select
                              value={stageIndex == null ? '' : String(stageIndex)}
                              onChange={e => setStageIndex(e.target.value === '' ? null : Number(e.target.value))}
                              dir="ltr"
                              data-testid="picker-stage-index"
                            >
                              {chosen.part.inspection_stage === 'IN_PROCESS' && <option value="">{t('pickerBendsInfer', lang)}</option>}
                              {chosen.part.inspection_stage !== 'IN_PROCESS' && <option value="">—</option>}
                              {Array.from({ length: (chosen.part.bends_total ?? 0) + 1 }, (_, k) => (
                                <option key={k} value={k}>
                                  {k}
                                  {chosen.part.bends_total != null ? ` / ${chosen.part.bends_total}` : ''}
                                </option>
                              ))}
                            </select>
                          </label>
                        )}
                      </div>
                    )}
                  </>
                )}

                {step === 'pick' && !scan.resolvable && (
                  <p className="st-picker-note is-warn" data-testid="picker-not-resolvable">{t('pickerNotResolvable', lang)}</p>
                )}

                {step === 'confirm' && chosen && (
                  <div className="st-picker-confirm" data-testid="picker-confirm">
                    <h3>{t('pickerConfirmTitle', lang)}</h3>
                    <p className="st-picker-confirm-part">
                      <bdi dir="ltr" className="st-picker-sku" data-testid="confirm-sku">{chosen.part.sku}</bdi>
                      {chosen.part.revision && (
                        <span dir="ltr">
                          {t('pickerRevision', lang)} <strong data-testid="confirm-revision">{chosen.part.revision}</strong>
                        </span>
                      )}
                      <span dir="ltr" data-testid="confirm-version">
                        v{chosen.part.versions.find(v => v.recipe_id === recipeId)?.recipe_version ?? chosen.part.recipe_version}
                      </span>
                    </p>
                    {chosen.part.part_name && <p dir="auto" className="st-picker-name">{chosen.part.part_name}</p>}
                    <p data-testid="confirm-stage">
                      {t('pickerStage', lang)}: <strong>{stage === 'FINAL' ? t('stageFinal', lang) : t('stageInProcess', lang)}</strong>
                      {stageIndex != null && (
                        <>
                          {' '}· {t('pickerBendsFormed', lang)}: <span className="st-num" dir="ltr">{stageIndex}</span>
                        </>
                      )}
                    </p>
                    <p className="st-picker-note">{t('pickerConfirmNote', lang)}</p>
                    {operatorBlock}
                  </div>
                )}

                {step === 'dismiss' && (
                  <div className="st-picker-dismiss" data-testid="picker-dismiss">
                    <h3>{t('pickerDismissTitle', lang)}</h3>
                    <fieldset>
                      <legend>{t('pickerDismissReason', lang)}</legend>
                      {DISMISS_REASONS.map(r => (
                        <label key={r} className={`st-picker-choice${dismissReason === r ? ' is-selected' : ''}`}>
                          <input type="radio" name="dismiss-reason" value={r} checked={dismissReason === r} onChange={() => setDismissReason(r)} />
                          <span>{dismissReasonText(r, lang)}</span>
                        </label>
                      ))}
                    </fieldset>
                    <label className="st-picker-notefield">
                      <span>
                        {t('pickerDismissNote', lang)} {dismissReason === 'OTHER' && <em>({t('pickerDismissNoteHint', lang)})</em>}
                      </span>
                      <textarea dir="auto" maxLength={500} value={dismissNote} onChange={e => setDismissNote(e.target.value)} rows={2} />
                    </label>
                    {operatorBlock}
                  </div>
                )}

                {step === 'done' && (
                  <p className="st-picker-done" role="status" data-testid="picker-done" data-result={done ?? ''}>
                    <CheckCircle2 size={30} aria-hidden /> {done === 'dismissed' ? t('pickerDismissed', lang) : t('pickerAssigned', lang)}
                  </p>
                )}
                {error && (
                  <p className="st-picker-error" role="alert" data-testid="picker-error">
                    {error}
                  </p>
                )}
              </section>
            </>
          )}
        </div>

        {scan && step !== 'done' && (
          <footer className="st-picker-foot">
            {step === 'pick' && (
              <button type="button" className="st-picker-btn is-secondary" onClick={() => { setError(null); setStep('dismiss') }} data-testid="picker-to-dismiss">
                {t('pickerDismiss', lang)}
              </button>
            )}
            {step !== 'pick' && (
              <button type="button" className="st-picker-btn is-secondary" onClick={() => { setError(null); setStep('pick') }} data-testid="picker-back">
                {t('pickerBack', lang)}
              </button>
            )}
            <span className="st-picker-spacer" />
            {step === 'pick' && scan.resolvable && (
              <button
                type="button"
                className="st-picker-btn is-primary"
                disabled={!chosen || !recipeId || !stageChoiceComplete(chosen.part, stage, stageIndex)}
                onClick={() => setStep('confirm')}
                data-testid="picker-next"
              >
                {t('pickerNext', lang)}
              </button>
            )}
            {step === 'confirm' && (
              <button type="button" className="st-picker-btn is-primary" disabled={busy || !mayOperate} onClick={() => void confirmAssign()} data-testid="picker-confirm-run">
                {busy ? t('pickerAssigning', lang) : t('pickerConfirm', lang)}
              </button>
            )}
            {step === 'dismiss' && (
              <button
                type="button"
                className="st-picker-btn is-danger"
                disabled={busy || !mayOperate || !canDismiss(dismissReason, dismissNote)}
                onClick={() => void confirmDismiss()}
                data-testid="picker-dismiss-do"
              >
                {t('pickerDismissDo', lang)}
              </button>
            )}
          </footer>
        )}
      </div>
    </div>
  )
}
