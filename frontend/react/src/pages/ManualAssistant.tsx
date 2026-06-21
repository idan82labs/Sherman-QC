import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  BookOpen,
  Bot,
  Camera,
  CheckCircle2,
  Clock,
  FileSearch,
  Languages,
  Loader2,
  Mic,
  RefreshCw,
  Send,
  Settings,
  ShieldAlert,
  X,
} from 'lucide-react'
import clsx from 'clsx'
import {
  getErrorMessage,
  manualAssistantApi,
  type ManualAttachmentResponse,
  type ManualChatResponse,
  type ManualEvidence,
  type ManualProfile,
} from '../services/api'

type UiLanguage = 'en' | 'he'

interface ChatItem {
  id: string
  role: 'user' | 'assistant'
  text: string
  response?: ManualChatResponse
  attachments?: ManualAttachmentResponse[]
  profile?: ManualProfile
  durationMs?: number
}

const copy = {
  en: {
    title: 'Manual Assistant',
    subtitle: 'Manual-grounded answers with cited pages and visual references.',
    cell: 'Cell Operation',
    software: 'Software',
    activeMode: 'Active mode',
    settings: 'Settings',
    uiLanguage: 'UI language',
    indexed: 'Indexed manuals',
    pages: 'pages',
    crops: 'crops',
    ready: 'Ready',
    render: 'Reindex',
    placeholder: 'Ask from the manual...',
    photo: 'Photo',
    voice: 'Voice',
    send: 'Send',
    evidence: 'Evidence',
    source: 'Source',
    openSource: 'Open source page',
    notFound: 'Not found in selected manuals',
    noEvidence: 'Citations appear here after an answer.',
    visualGap: 'Visual evidence gap',
    supported: 'Supported',
    clarification: 'Needs a manual question',
    switchTo: 'Switch to',
    ingesting: 'Indexing manuals...',
    typing: 'Retrieving manual evidence...',
    empty: 'Ask a procedure, safety, software, screenshot, or drawing question.',
    examples: 'Try',
    answerTime: 'Answer time',
    citations: 'citations',
    attachments: 'attachments',
    photoStored: 'Photo attached to this turn. Image reasoning is not enabled yet.',
    voiceNeedsKey: 'Voice transcription needs OpenAI integration; typed questions work now.',
  },
  he: {
    title: 'מסייע מדריכים',
    subtitle: 'תשובות מתוך המדריך עם עמודים ותמונות מקור.',
    cell: 'תפעול תא',
    software: 'תוכנה',
    activeMode: 'מצב פעיל',
    settings: 'הגדרות',
    uiLanguage: 'שפת ממשק',
    indexed: 'מדריכים מאונדקסים',
    pages: 'עמודים',
    crops: 'חיתוכים',
    ready: 'מוכן',
    render: 'אינדוקס',
    placeholder: 'שאל שאלה מהמדריך...',
    photo: 'תמונה',
    voice: 'קול',
    send: 'שלח',
    evidence: 'ראיות',
    source: 'מקור',
    openSource: 'פתח עמוד מקור',
    notFound: 'לא נמצא במדריכים שנבחרו',
    noEvidence: 'ציטוטים יופיעו כאן אחרי תשובה.',
    visualGap: 'חסר מקור חזותי',
    supported: 'נתמך',
    clarification: 'נדרשת שאלה מהמדריך',
    switchTo: 'עבור אל',
    ingesting: 'מאנדקס מדריכים...',
    typing: 'מחפש ראיות במדריך...',
    empty: 'שאל שאלה על פרוצדורה, בטיחות, תוכנה, צילום מסך או שרטוט.',
    examples: 'דוגמאות',
    answerTime: 'זמן תשובה',
    citations: 'ציטוטים',
    attachments: 'קבצים',
    photoStored: 'התמונה צורפה לשאלה הזו. ניתוח תמונה עדיין לא פעיל.',
    voiceNeedsKey: 'תמלול קולי דורש אינטגרציית OpenAI; שאלות טקסט עובדות כעת.',
  },
}

const examplePrompts: Record<UiLanguage, Record<ManualProfile, string[]>> = {
  en: {
    cell_operation: [
      'Who is allowed to open the electrical cabinet?',
      'How should BendMaster be stopped in an emergency?',
      'What is the ToolMaster used for?',
    ],
    software: [
      'How do I import a DXF file with bending lines?',
      'How do I create a rounded corner in 2D mode?',
      'Which settings are relevant to bending in 3D design?',
    ],
  },
  he: {
    cell_operation: [
      'מי רשאי לפתוח את ארון החשמל?',
      'איך עוצרים את BendMaster במצב חירום?',
      'למה משמש ה-ToolMaster?',
    ],
    software: [
      'איך מייבאים קובץ DXF עם קווי כיפוף?',
      'איך יוצרים פינה עגולה במצב 2D?',
      'אילו הגדרות רלוונטיות לכיפוף בתכנון 3D?',
    ],
  },
}

function formatMs(value?: number) {
  if (value === undefined) return ''
  if (value < 1000) return `${Math.round(value)} ms`
  return `${(value / 1000).toFixed(1)} s`
}

function supportLabel(lang: UiLanguage, response: ManualChatResponse) {
  if (response.support_state === 'clarification') return copy[lang].clarification
  if (response.support_state === 'not_found') return copy[lang].notFound
  if (response.support_state === 'partial_support_visual_gap') return copy[lang].visualGap
  return copy[lang].supported
}

function supportTone(response: ManualChatResponse) {
  if (response.support_state === 'clarification') return 'border-sky-500/30 text-sky-300 bg-sky-500/10'
  if (response.support_state === 'not_found') return 'border-amber-500/30 text-amber-300 bg-amber-500/10'
  if (response.support_state === 'partial_support_visual_gap') {
    return 'border-orange-500/30 text-orange-300 bg-orange-500/10'
  }
  return 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10'
}

export default function ManualAssistant() {
  const [uiLanguage, setUiLanguage] = useState<UiLanguage>(() => {
    return (localStorage.getItem('manualAssistantUiLanguage') as UiLanguage) || 'en'
  })
  const [profile, setProfile] = useState<ManualProfile>(() => {
    return (localStorage.getItem('manualAssistantProfile') as ManualProfile) || 'cell_operation'
  })
  const [message, setMessage] = useState('')
  const [items, setItems] = useState<ChatItem[]>([])
  const [attachments, setAttachments] = useState<ManualAttachmentResponse[]>([])
  const [isSending, setIsSending] = useState(false)
  const [isIngesting, setIsIngesting] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const conversationRef = useRef<HTMLDivElement | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])

  const t = copy[uiLanguage]
  const isRtl = uiLanguage === 'he'

  const manualsQuery = useQuery({
    queryKey: ['manual-assistant', 'manuals'],
    queryFn: manualAssistantApi.listManuals,
    retry: false,
  })

  useEffect(() => {
    localStorage.setItem('manualAssistantUiLanguage', uiLanguage)
    localStorage.setItem('manualAssistantProfile', profile)
  }, [uiLanguage, profile])

  useEffect(() => {
    conversationRef.current?.scrollTo({
      top: conversationRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [items, isSending])

  const handleIngest = async () => {
    setError('')
    setNotice('')
    setIsIngesting(true)
    try {
      await manualAssistantApi.ingest(true)
      await manualsQuery.refetch()
      setNotice('Manual index is ready.')
    } catch (err: unknown) {
      setError(getErrorMessage(err))
    } finally {
      setIsIngesting(false)
    }
  }

  const handlePhoto = async (file: File) => {
    setError('')
    setNotice('')
    try {
      const uploaded = await manualAssistantApi.uploadPhoto(file)
      setAttachments((current) => [...current, uploaded])
      setNotice(t.photoStored)
    } catch (err: unknown) {
      setError(getErrorMessage(err))
    }
  }

  const handleSend = async () => {
    const trimmed = message.trim()
    if (!trimmed && attachments.length === 0) return

    setError('')
    setNotice('')
    setIsSending(true)
    const startedAt = performance.now()
    const turnAttachments = attachments
    const userText = trimmed || '[photo]'
    setItems((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: 'user',
        text: userText,
        attachments: turnAttachments,
        profile,
      },
    ])
    setMessage('')
    setAttachments([])

    try {
      const response = await manualAssistantApi.chat({
        profile,
        message: userText,
        attachment_ids: turnAttachments.map((item) => item.attachment_id),
        ui_language: uiLanguage,
        answer_language: 'follow_ui',
        retrieval_profile: 'accurate',
      })
      setItems((current) => [
        ...current,
        {
          id: response.request_id,
          role: 'assistant',
          text: response.answer,
          response,
          profile,
          durationMs: performance.now() - startedAt,
        },
      ])
    } catch (err: unknown) {
      setError(getErrorMessage(err))
    } finally {
      setIsSending(false)
    }
  }

  const startRecording = async () => {
    setError('')
    setNotice('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setNotice(t.voiceNeedsKey)
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      chunksRef.current = []
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        try {
          const result = await manualAssistantApi.transcribe(blob)
          setMessage((current) => `${current}${current ? ' ' : ''}${result.text}`)
        } catch {
          setNotice(t.voiceNeedsKey)
        } finally {
          setIsRecording(false)
        }
      }
      recorder.start()
      setIsRecording(true)
    } catch {
      setNotice(t.voiceNeedsKey)
    }
  }

  const stopRecording = () => {
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop()
    }
  }

  const totalPages = manualsQuery.data?.total_pages || 0
  const cropCount = manualsQuery.data?.crop_count || 0
  const latestResponseItem = [...items].reverse().find((item) => item.response)
  const latestEvidence = latestResponseItem?.response?.citations || []
  const activeExamples = examplePrompts[uiLanguage][profile]

  return (
    <div className={clsx('min-h-full', isRtl && 'rtl')} dir={isRtl ? 'rtl' : 'ltr'}>
      <div className="mx-auto flex max-w-7xl flex-col gap-3 lg:h-[calc(100vh-3rem)]">
        <header className="grid gap-3 border-b border-dark-700 pb-3 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-center">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-3">
              <Bot className="h-7 w-7 text-primary-400" />
              <h1 className="text-2xl font-bold text-dark-100">{t.title}</h1>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-300">
                <CheckCircle2 className="h-3.5 w-3.5" />
                {t.ready}
              </span>
            </div>
            <p className="mt-1 max-w-3xl text-sm text-dark-400">{t.subtitle}</p>
          </div>

          <div className="flex flex-wrap items-center gap-2 xl:justify-end">
            <div className="flex rounded-md border border-dark-600 bg-dark-800 p-1" aria-label={t.activeMode}>
              {(['cell_operation', 'software'] as ManualProfile[]).map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setProfile(value)}
                  className={clsx(
                    'rounded px-3 py-2 text-sm font-medium transition-colors',
                    profile === value
                      ? 'bg-primary-500 text-white'
                      : 'text-dark-300 hover:bg-dark-700 hover:text-dark-100'
                  )}
                  aria-pressed={profile === value}
                >
                  {value === 'cell_operation' ? t.cell : t.software}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2 rounded-md border border-dark-600 bg-dark-800 px-3 py-2" aria-label={t.uiLanguage}>
              <Languages className="h-4 w-4 text-dark-400" />
              <button
                type="button"
                onClick={() => setUiLanguage('en')}
                className={clsx('min-h-8 min-w-9 rounded px-2 text-sm', uiLanguage === 'en' ? 'text-primary-300' : 'text-dark-300 hover:bg-dark-700')}
                aria-pressed={uiLanguage === 'en'}
              >
                EN
              </button>
              <span className="text-dark-600">/</span>
              <button
                type="button"
                onClick={() => setUiLanguage('he')}
                className={clsx('min-h-8 min-w-9 rounded px-2 text-sm', uiLanguage === 'he' ? 'text-primary-300' : 'text-dark-300 hover:bg-dark-700')}
                aria-pressed={uiLanguage === 'he'}
              >
                HE
              </button>
            </div>
          </div>
        </header>

        <section className="grid gap-3 border-b border-dark-700 pb-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-dark-300">
            <span className="inline-flex items-center gap-2 text-dark-200">
              <FileSearch className="h-4 w-4 text-primary-400" />
              {t.indexed}: {manualsQuery.data?.manuals?.length || 0}
            </span>
            <span>{totalPages} {t.pages}</span>
            <span>{cropCount} {t.crops}</span>
            {latestResponseItem?.durationMs !== undefined && (
              <span className="inline-flex items-center gap-1.5">
                <Clock className="h-4 w-4 text-dark-500" />
                {t.answerTime}: {formatMs(latestResponseItem.durationMs)}
              </span>
            )}
            {manualsQuery.isLoading && <span>{t.ingesting}</span>}
          </div>
          <button
            type="button"
            onClick={handleIngest}
            disabled={isIngesting}
            className="btn btn-secondary inline-flex items-center justify-center gap-2 rounded-md"
          >
            {isIngesting ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
            {t.render}
          </button>
        </section>

        {(error || notice) && (
          <div
            className={clsx(
              'rounded-lg border px-4 py-3 text-sm',
              error ? 'border-red-500/30 bg-red-500/10 text-red-300' : 'border-primary-500/30 bg-primary-500/10 text-primary-200'
            )}
          >
            {error || notice}
          </div>
        )}

        <main className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_400px]">
          <section className="flex min-h-[68vh] flex-col overflow-hidden rounded-md border border-dark-700 bg-dark-800/60 lg:min-h-0">
            <div ref={conversationRef} className="flex-1 space-y-4 overflow-auto p-3 sm:p-4">
              {items.length === 0 && (
                <div className="flex h-full min-h-[360px] flex-col items-center justify-center text-center text-dark-400">
                  <BookOpen className="mb-4 h-12 w-12 text-dark-600" />
                  <p className="max-w-md text-sm leading-6">{t.empty}</p>
                  <div className="mt-5 w-full max-w-2xl">
                    <div className="mb-2 text-xs font-medium uppercase tracking-wide text-dark-500">{t.examples}</div>
                    <div className="flex flex-wrap justify-center gap-2">
                      {activeExamples.map((example) => (
                        <button
                          key={example}
                          type="button"
                          onClick={() => setMessage(example)}
                          className="rounded-md border border-dark-600 bg-dark-900/80 px-3 py-2 text-left text-xs leading-5 text-dark-200 transition-colors hover:border-primary-500/50 hover:text-primary-200"
                        >
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {items.map((item) => (
                <div
                  key={item.id}
                  className={clsx(
                    'flex',
                    item.role === 'user' ? (isRtl ? 'justify-start' : 'justify-end') : (isRtl ? 'justify-end' : 'justify-start')
                  )}
                >
                  <div
                    className={clsx(
                      'max-w-[860px] rounded-md border px-4 py-3',
                      item.role === 'user'
                        ? 'border-primary-500/30 bg-primary-500/15 text-dark-100'
                        : 'border-dark-600 bg-dark-900/70 text-dark-100'
                    )}
                  >
                    <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                      {item.profile && (
                        <span className="rounded bg-dark-700 px-2 py-1 text-dark-300">
                          {item.profile === 'cell_operation' ? t.cell : t.software}
                        </span>
                      )}
                      {item.response && (
                        <span className={clsx('inline-flex items-center gap-2 rounded border px-2 py-1', supportTone(item.response))}>
                          {item.response.support_state === 'supported' ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
                          {supportLabel(uiLanguage, item.response)}
                        </span>
                      )}
                      {item.durationMs !== undefined && (
                        <span className="inline-flex items-center gap-1 rounded bg-dark-700 px-2 py-1 text-dark-300">
                          <Clock className="h-3.5 w-3.5" />
                          {formatMs(item.durationMs)}
                        </span>
                      )}
                    </div>
                    <p className="whitespace-pre-wrap text-sm leading-6">{item.text}</p>
                    {item.attachments && item.attachments.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {item.attachments.map((attachment) => (
                          <span key={attachment.attachment_id} className="rounded-md bg-dark-700 px-2 py-1 text-xs text-dark-300">
                            {attachment.filename}
                          </span>
                        ))}
                      </div>
                    )}
                    {item.response?.warnings?.map((warning) => (
                      <div key={warning} className="mt-3 flex items-start gap-2 rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                        <ShieldAlert className="mt-0.5 h-4 w-4 flex-shrink-0" />
                        <span>{warning}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              {isSending && (
                <div className="flex items-center gap-2 text-sm text-dark-400">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {t.typing}
                </div>
              )}
            </div>

            <div className="border-t border-dark-700 p-3">
              {attachments.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-2">
                  {attachments.map((attachment) => (
                    <span key={attachment.attachment_id} className="inline-flex items-center gap-2 rounded-md bg-dark-700 px-2 py-1 text-xs text-dark-200">
                      {attachment.filename}
                      <button
                        type="button"
                        onClick={() => setAttachments((current) => current.filter((item) => item.attachment_id !== attachment.attachment_id))}
                        title="Remove"
                        aria-label={`Remove ${attachment.filename}`}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
              <div className="flex items-end gap-2">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (file) void handlePhoto(file)
                    event.target.value = ''
                  }}
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="btn btn-secondary flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-md p-0"
                  title={t.photo}
                  aria-label={t.photo}
                >
                  <Camera className="h-5 w-5" />
                </button>
                <button
                  type="button"
                  onClick={isRecording ? stopRecording : startRecording}
                  className={clsx('btn flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-md p-0', isRecording ? 'btn-danger' : 'btn-secondary')}
                  title={t.voice}
                  aria-label={t.voice}
                >
                  <Mic className="h-5 w-5" />
                </button>
                <textarea
                  value={message}
                  onChange={(event) => setMessage(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      void handleSend()
                    }
                  }}
                  placeholder={t.placeholder}
                  className="input min-h-[52px] min-w-0 flex-1 resize-none rounded-md"
                  rows={1}
                />
                <button
                  type="button"
                  onClick={handleSend}
                  disabled={isSending || (!message.trim() && attachments.length === 0)}
                  className="btn btn-primary flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-md p-0 disabled:opacity-50"
                  title={t.send}
                  aria-label={t.send}
                >
                  {isSending ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
                </button>
              </div>
            </div>
          </section>

          <aside className="min-h-[360px] overflow-auto rounded-md border border-dark-700 bg-dark-800/60 p-4 lg:min-h-0">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-semibold text-dark-100">
                <Settings className="h-5 w-5 text-dark-400" />
                {t.evidence}
              </h2>
              {latestEvidence.length > 0 && (
                <span className="rounded bg-dark-700 px-2 py-1 text-xs text-dark-300">
                  {latestEvidence.length} {t.citations}
                </span>
              )}
            </div>
            <EvidencePanel evidence={latestEvidence} language={uiLanguage} />
          </aside>
        </main>
      </div>
    </div>
  )
}

function EvidencePanel({ evidence, language }: { evidence: ManualEvidence[]; language: UiLanguage }) {
  const t = copy[language]
  if (evidence.length === 0) {
    return (
      <div className="flex min-h-52 flex-col items-center justify-center rounded-md border border-dashed border-dark-700 bg-dark-900/40 px-4 text-center">
        <FileSearch className="mb-3 h-8 w-8 text-dark-600" />
        <p className="max-w-xs text-sm leading-6 text-dark-400">{t.noEvidence}</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {evidence.map((item) => {
        const imageUrl = item.crop?.url || item.page_image?.url
        const score = Number(item.retrieval.score)
        return (
          <article key={item.citation_id} className="rounded-md border border-dark-700 bg-dark-900/60 p-3">
            <div className="mb-2 flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-dark-100">{item.manual_id}</p>
                <p className="text-xs text-dark-400">p.{item.page_number} · {item.element_type}</p>
              </div>
              <span className="rounded bg-dark-700 px-2 py-1 text-xs text-dark-300">
                {Number.isFinite(score) ? score.toFixed(1) : '-'}
              </span>
            </div>
            {imageUrl && (
              <a href={imageUrl} target="_blank" rel="noreferrer" className="mb-3 block overflow-hidden rounded border border-dark-700 bg-dark-950">
                <img src={imageUrl} alt={`${item.manual_id} page ${item.page_number}`} className="max-h-56 w-full object-contain" />
              </a>
            )}
            <p className="text-xs leading-5 text-dark-300">{item.excerpt}</p>
            <a href={item.page_image?.url || imageUrl || '#'} target="_blank" rel="noreferrer" className="mt-3 inline-flex min-h-9 items-center gap-1.5 rounded px-1 text-xs font-medium text-primary-300 hover:text-primary-200">
              <BookOpen className="h-3.5 w-3.5" />
              {t.openSource}
            </a>
          </article>
        )
      })}
    </div>
  )
}
