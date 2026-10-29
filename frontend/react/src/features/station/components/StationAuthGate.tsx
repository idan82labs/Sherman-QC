/**
 * Station token entry. Shown when no usable token exists, or after the
 * server rejected the saved one (401) or refused this station (403).
 * The token is stored per station in localStorage (inside try/catch); it is
 * never placed in a URL.
 */
import { useState, type FormEvent } from 'react'

import { DEVICE_TOKEN_RE } from '../api/stationApi'
import { t } from '../copy'
import type { Lang } from '../model/types'

export type AuthGateReason = 'missing' | 'rejected' | 'forbidden'

export interface StationAuthGateProps {
  lang: Lang
  stationId: string
  reason: AuthGateReason
  onSubmit: (token: string) => void
  onToggleLang: () => void
}

export function StationAuthGate({ lang, stationId, reason, onSubmit, onToggleLang }: StationAuthGateProps) {
  const [value, setValue] = useState('')
  const [formatError, setFormatError] = useState(false)

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const token = value.trim()
    if (!DEVICE_TOKEN_RE.test(token)) {
      setFormatError(true)
      return
    }
    setFormatError(false)
    onSubmit(token)
  }

  return (
    <div className="st-gate" data-testid="station-auth-gate" data-reason={reason}>
      <form className="st-gate-card" onSubmit={submit}>
        <div className="st-gate-top">
          <h1>{t('authTitle', lang)}</h1>
          <button type="button" className="st-lang" onClick={onToggleLang} aria-label={t('langToggleLabel', lang)}>
            {t('langToggle', lang)}
          </button>
        </div>
        <p className="st-gate-station">
          {t('station', lang)} <strong dir="ltr">{stationId}</strong>
        </p>
        <p>{reason === 'missing' ? t('authBody', lang) : reason === 'forbidden' ? t('authForbidden', lang) : t('authRejected', lang)}</p>
        <label className="st-gate-label" htmlFor="st-token">{t('authTokenLabel', lang)}</label>
        <input
          id="st-token"
          className="st-gate-input"
          type="password"
          autoComplete="off"
          spellCheck={false}
          dir="ltr"
          value={value}
          onChange={e => setValue(e.target.value)}
          aria-invalid={formatError}
        />
        {formatError && <p className="st-gate-error">{t('authInvalidFormat', lang)}</p>}
        <button type="submit" className="st-gate-submit">{t('authConnect', lang)}</button>
      </form>
    </div>
  )
}
