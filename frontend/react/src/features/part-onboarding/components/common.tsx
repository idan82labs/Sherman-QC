/**
 * Small shared pieces of the onboarding UI: language state, the language
 * switch, notices, status chips, reason rows and the file drop.
 */
import { Fragment, useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, Ban, CheckCircle2, ChevronDown, Info, Upload } from 'lucide-react'
import { persistVendorLanguage, readVendorLanguage } from '../../part-catalog/pages/vendorSetupCopy'
import { fill, onbCopy, reasonFor, type OnbCopy, type OnbLanguage } from '../copy'
import type { StepStatus } from '../model/checklist'
import type { ProductionApiError } from '../model/types'

/** Language shared with Vendor Setup (same storage key); sets <html lang/dir> while mounted. */
export function useOnboardingLanguage(): [OnbLanguage, (next: OnbLanguage) => void] {
  const [language, setLanguage] = useState<OnbLanguage>(() => readVendorLanguage())
  useEffect(() => {
    persistVendorLanguage(language)
    const root = document.documentElement
    const original = { lang: root.lang, dir: root.dir }
    root.lang = language
    root.dir = language === 'he' ? 'rtl' : 'ltr'
    return () => { root.lang = original.lang; root.dir = original.dir }
  }, [language])
  return [language, setLanguage]
}

export function LanguageToggle({ language, onChange }: { language: OnbLanguage; onChange: (next: OnbLanguage) => void }) {
  const t = onbCopy[language]
  return (
    <div className="onb-toggle" role="group" aria-label={t.language}>
      {(['he', 'en'] as const).map(option => (
        <button key={option} type="button" lang={option} dir={option === 'he' ? 'rtl' : 'ltr'} aria-pressed={language === option}
          data-language={option} onClick={() => onChange(option)}>
          {option === 'he' ? t.hebrew : t.english}
        </button>
      ))}
    </div>
  )
}

export type NoticeTone = 'info' | 'attention' | 'blocked' | 'neutral'

export function Notice({ tone = 'neutral', children, role }: { tone?: NoticeTone; children: ReactNode; role?: 'status' | 'alert' }) {
  const Icon = tone === 'blocked' ? Ban : tone === 'attention' ? AlertTriangle : tone === 'info' ? Info : Info
  const color = tone === 'blocked' ? 'var(--onb-red)' : tone === 'attention' ? 'var(--onb-amber)' : tone === 'info' ? 'var(--onb-dykem)' : 'var(--onb-muted)'
  return <div className="onb-note" data-tone={tone} role={role}><Icon size={18} color={color} aria-hidden /><div className="min-w-0">{children}</div></div>
}

export function StatusChip({ status, t }: { status: StepStatus; t: OnbCopy }) {
  const tone = status === 'done' ? 'done' : status === 'attention' || status === 'working' ? 'attention' : status === 'blocked' ? 'blocked' : undefined
  return <span className="onb-chip" data-tone={tone} data-status={status}>{status === 'done' && <CheckCircle2 size={14} aria-hidden />}{t[`status${status}` as const]}</span>
}

/**
 * One code in plain language: what is wrong, then what to do. The raw code is
 * kept for support but sits quietly at the end (never the primary text).
 * `quiet` renders an informational code (nothing to fix) without a severity edge.
 */
export function ReasonRow({ code, language, severity = 'BLOCKER', serverMessage, detail, quiet, action }: {
  code: string; language: OnbLanguage; severity?: 'BLOCKER' | 'WARNING'; serverMessage?: string; detail?: string | null; quiet?: boolean; action?: ReactNode
}) {
  const reason = reasonFor(code, language, serverMessage)
  return (
    <div className="onb-reason" data-severity={quiet ? 'INFO' : severity} data-reason-code={code}>
      <div className="onb-reason-text">
        <strong>{reason.title}</strong>
        <span className="onb-reason-action">{reason.action}</span>
        {detail && <span className="onb-help" dir="auto">{detail}</span>}
        <span className="onb-reason-code" dir="ltr" title={code}>{code}</span>
      </div>
      {action && <div className="onb-reason-cta">{action}</div>}
    </div>
  )
}

/** Several reasons: the first in full, the rest behind "N more" (progressive disclosure). */
export function ReasonList({ codes, language, severity, quiet, t, render }: {
  codes: string[]; language: OnbLanguage; severity?: 'BLOCKER' | 'WARNING'; quiet?: boolean; t: OnbCopy
  render?: (code: string) => ReactNode
}) {
  if (!codes.length) return null
  const row = (code: string) => render ? render(code) : <ReasonRow key={code} code={code} language={language} severity={severity} quiet={quiet} />
  const [first, ...rest] = codes
  return (
    <div className="grid gap-2">
      {row(first)}
      {rest.length > 0 && <Disclosure summary={fill(t.moreReasons, { n: rest.length })} compact>
        <div className="grid gap-2">{rest.map(row)}</div>
      </Disclosure>}
    </div>
  )
}

/** A quiet, keyboard-friendly <details> with a rotating chevron. */
export function Disclosure({ summary, children, compact, defaultOpen, ...rest }: {
  summary: ReactNode; children: ReactNode; compact?: boolean; defaultOpen?: boolean; [data: `data-${string}`]: string | undefined
}) {
  return (
    <details className="onb-disclosure" data-compact={compact ? 'true' : undefined} open={defaultOpen} {...rest}>
      <summary><ChevronDown size={16} aria-hidden className="onb-disclosure-chevron" /><span>{summary}</span></summary>
      <div className="onb-disclosure-body">{children}</div>
    </details>
  )
}

/** A friendly empty state: what this is, why it matters, and the one thing to do. */
export function EmptyState({ icon, title, children, actions, ...rest }: {
  icon?: ReactNode; title: string; children?: ReactNode; actions?: ReactNode; [data: `data-${string}`]: string | undefined
}) {
  return (
    <div className="onb-empty" {...rest}>
      {icon && <span className="onb-empty-icon" aria-hidden>{icon}</span>}
      <div className="grid gap-2">
        <h2 className="onb-empty-title">{title}</h2>
        {children && <div className="onb-lead">{children}</div>}
        {actions && <div className="onb-actions mt-2">{actions}</div>}
      </div>
    </div>
  )
}

export function ApiErrorNotice({ error, language }: { error: ProductionApiError | null | undefined; language: OnbLanguage }) {
  if (!error) return null
  return (
    <div role="alert" className="grid gap-2" data-api-error={error.status}>
      {error.codes.length ? error.codes.map(code => <ReasonRow key={code} code={code} language={language} serverMessage={error.messages[code] || error.message} />)
        : <Notice tone="blocked">{error.message || reasonFor('SERVER_ERROR', language).title}</Notice>}
    </div>
  )
}

/** Human file size (1.2 MB). */
export function formatBytes(bytes: number | null | undefined): string {
  if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function extensionOf(name: string | null | undefined): string {
  const match = /\.([a-z0-9]{2,5})$/i.exec(name || '')
  return match ? match[1].toUpperCase() : ''
}

/**
 * A file the part already has (or was just given): a typed tag, the name, quiet
 * facts, and the parse result. `result` is the one-line outcome (for example
 * "4 bends found in the CAD").
 */
export function FileCard({ name, kind, facts, result, resultTone = 'done', actions, ...rest }: {
  name: string; kind?: string; facts?: ReactNode; result?: ReactNode; resultTone?: 'done' | 'attention' | 'blocked'
  actions?: ReactNode; [data: `data-${string}`]: string | undefined
}) {
  const ext = (kind || extensionOf(name) || 'FILE').slice(0, 4)
  return (
    <div className="onb-filecard" {...rest}>
      <span className="onb-filetag" aria-hidden><span>{ext}</span></span>
      <div className="grid min-w-0 gap-0.5">
        <span className="onb-file-name" dir="ltr">{name}</span>
        {facts && <span className="onb-help">{facts}</span>}
        {result && <span className="onb-result" data-tone={resultTone}>
          {resultTone === 'done' ? <CheckCircle2 size={16} aria-hidden /> : <AlertTriangle size={16} aria-hidden />}{result}</span>}
      </div>
      {actions && <div className="onb-filecard-actions">{actions}</div>}
    </div>
  )
}

/**
 * Drag-and-drop upload zone. Empty: a large target with the title and a
 * "Choose file" button (the step's primary action). With a file already there
 * (`compact`): a slim strip to replace it. While uploading it shows the file
 * being sent and its progress.
 */
export function FileDrop({ accept, disabled, busy, progress, label, hint, onFile, t, title, compact }: {
  accept: string; disabled?: boolean; busy?: boolean; progress?: number | null; label: string; hint: string
  onFile: (file: File) => void; t: OnbCopy; title?: string; compact?: boolean
}) {
  const inputId = useId()
  const [over, setOver] = useState(false)
  const [sending, setSending] = useState<File | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const take = (file: File) => { setSending(file); onFile(file) }
  const off = disabled || busy
  return (
    <div className="onb-drop" data-over={over} data-compact={compact ? 'true' : 'false'} data-busy={busy ? 'true' : 'false'}
      onDragOver={event => { if (off) return; event.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={event => {
        event.preventDefault(); setOver(false)
        const file = event.dataTransfer.files?.[0]
        if (file && !off) take(file)
      }}>
      {!compact && <span className="onb-drop-icon" aria-hidden><Upload size={22} /></span>}
      <div className="onb-drop-text">
        {busy && sending ? <>
          <strong className="onb-drop-title"><bdi dir="ltr">{sending.name}</bdi></strong>
          <span className="onb-help onb-num">{formatBytes(sending.size)}</span>
        </> : <>
          <strong className="onb-drop-title">{over ? t.dropRelease : compact ? t.dropReplace : title || t.dropHint}</strong>
          <span className="onb-help">{hint}</span>
        </>}
      </div>
      {!busy && <div className="onb-drop-cta">
        {!compact && <span className="onb-help">{t.dropOr}</span>}
        <label htmlFor={inputId} className={`onb-btn ${compact ? 'onb-btn-quiet' : 'onb-btn-primary'} cursor-pointer`} aria-disabled={off}>
          <Upload size={18} aria-hidden />{label}
        </label>
      </div>}
      <input ref={inputRef} id={inputId} className="sr-only" type="file" accept={accept} disabled={off}
        onChange={event => { const file = event.target.files?.[0]; if (file) take(file); event.target.value = '' }} />
      {busy && progress != null && <div className="onb-drop-progress">
        <div className="onb-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><span style={{ width: `${progress}%` }} /></div>
        {/* 100 % sent: the server is still storing / analysing the file, so say that instead of "Uploading… 100%". */}
        <span className="onb-help onb-num" data-upload-phase={progress >= 100 ? 'processing' : 'sending'}>
          {progress >= 100 ? t.uploadProcessing : fill(t.uploading, { p: progress })}</span>
      </div>}
    </div>
  )
}

/** `fill` for templates whose values are elements ({n} -> <bdi>4</bdi>). */
export function fillNodes(template: string, values: Record<string, ReactNode>): ReactNode[] {
  return template.split(/(\{[a-z]+\})/i).map((part, index) => {
    const match = /^\{([a-z]+)\}$/i.exec(part)
    return <Fragment key={index}>{match && match[1] in values ? values[match[1]] : part}</Fragment>
  })
}

export function shortHash(hash: string | null | undefined): string {
  return hash ? hash.slice(0, 12) : ''
}

export function fileNameOf(path: string | null | undefined): string {
  if (!path) return ''
  return path.split(/[\\/]/).pop() || path
}

const FILE_NAMES_KEY = 'sherman.onboarding.fileNames'

function readFileNames(): Record<string, string> {
  try {
    const raw = window.localStorage.getItem(FILE_NAMES_KEY)
    const parsed = raw ? JSON.parse(raw) as unknown : null
    return parsed && typeof parsed === 'object' ? parsed as Record<string, string> : {}
  } catch {
    return {}
  }
}

/**
 * The catalog stores uploads under generated names (`9598e8b3….step`). The name the
 * operator chose is remembered in this browser, keyed by the file's SHA-256 (the
 * server's file fingerprint), so the wizard can show it. A convenience only: another
 * browser falls back to `displayFileName`'s neutral label.
 */
export async function rememberFileName(file: File): Promise<void> {
  try {
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
    const hex = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')
    const names = readFileNames()
    names[hex] = file.name
    const keys = Object.keys(names)
    for (const key of keys.slice(0, Math.max(0, keys.length - 200))) delete names[key]
    window.localStorage.setItem(FILE_NAMES_KEY, JSON.stringify(names))
  } catch {
    // no storage / no crypto: the neutral label is shown
  }
}

/** The uploaded name for a stored file: remembered name, else the stored name unless it is a generated one. */
export function displayFileName(sha256: string | null | undefined, storedPath: string | null | undefined, fallback: string): string {
  const remembered = sha256 ? readFileNames()[sha256.toLowerCase()] : undefined
  if (remembered) return remembered
  const stored = fileNameOf(storedPath)
  return !stored || /^[0-9a-f]{24,}(\.[a-z0-9]+)?$/i.test(stored) ? fallback : stored
}
