import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import {
  AlertTriangle,
  ArrowUp,
  Bot,
  Camera,
  CheckCircle2,
  ChevronDown,
  ClipboardList,
  FileText,
  Languages,
  Loader2,
  Mic,
  MicOff,
  Paperclip,
  Plus,
  Search,
  Settings,
  ShieldCheck,
  X,
  Zap,
} from 'lucide-react'
import {
  chatGptAuthApi,
  getErrorMessage,
  healthApi,
  manualAssistantApi,
  type ChatGptSessionResponse,
  type ManualAttachmentResponse,
  type ManualChatResponse,
  type ManualEvidence,
  type ManualProfile,
} from '../services/api'

type ChatRole = 'user' | 'assistant'
type ChatMode = 'ask' | 'troubleshoot' | 'procedure' | 'quality' | 'summarize'
type ContextMode = 'auto' | ManualProfile
type UiLanguage = 'en' | 'he'
type VoiceLanguage = 'en-US' | 'he-IL' | 'ru-RU'

const SHERMAN_LOGO_SRC = '/sherman-logo.png'

interface SpeechRecognitionAlternativeLike {
  transcript: string
}

interface SpeechRecognitionResultLike {
  isFinal: boolean
  [index: number]: SpeechRecognitionAlternativeLike | undefined
}

interface SpeechRecognitionEventLike {
  resultIndex: number
  results: ArrayLike<SpeechRecognitionResultLike>
}

interface SpeechRecognitionLike {
  lang: string
  continuous: boolean
  interimResults: boolean
  onresult: ((event: SpeechRecognitionEventLike) => void) | null
  onerror: (() => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike
type SpeechWindow = Window &
  typeof globalThis & {
    SpeechRecognition?: SpeechRecognitionConstructor
    webkitSpeechRecognition?: SpeechRecognitionConstructor
  }

interface ChatTurn {
  id: string
  role: ChatRole
  text: string
  response?: ManualChatResponse
  durationMs?: number
  attachments?: ManualAttachmentResponse[]
}

const modeIcons: Record<ChatMode, typeof Search> = {
  ask: Search,
  troubleshoot: Zap,
  procedure: ClipboardList,
  quality: ShieldCheck,
  summarize: FileText,
}

const modeLabels: Record<UiLanguage, Record<ChatMode, string>> = {
  en: {
    ask: 'Ask',
    troubleshoot: 'Troubleshoot',
    procedure: 'Procedure',
    quality: 'Quality',
    summarize: 'Summarize',
  },
  he: {
    ask: 'שאלה',
    troubleshoot: 'תקלה',
    procedure: 'נוהל',
    quality: 'איכות',
    summarize: 'סיכום',
  },
}

const copy = {
  en: {
    appName: 'ShermanAI',
    productName: 'ShermanChat',
    settings: 'Settings',
    newChat: 'New chat',
    uiLanguage: 'Interface',
    voiceLanguage: 'Voice',
    english: 'English',
    hebrew: 'עברית',
    auto: 'Auto',
    cell: 'Cell',
    software: 'Software',
    attachPhoto: 'Attach photo',
    startVoice: 'Start voice transcript',
    stopVoice: 'Stop voice transcript',
    voiceUnavailable: 'Voice transcript is not available in this browser.',
    voiceError: 'Voice transcript stopped before text was captured.',
    placeholder: 'Ask ShermanAI...',
    send: 'Send',
    thinking: 'Thinking...',
    sources: 'Sources used',
    notFound: 'No approved source',
    needsDetail: 'Needs detail',
    chat: 'Chat',
    model: 'Model',
    mock: 'mock',
    removeAttachment: 'Remove attachment',
    introTitle: 'How can I help with the cell today?',
    introSubtitle: 'Ask naturally. ShermanAI searches manuals only when the answer needs approved evidence.',
    photoReady: 'Photo attached',
    connectChatGpt: 'Connect ChatGPT',
    disconnectChatGpt: 'Disconnect',
    chatGptConnected: 'ChatGPT connected',
    chatGptRequired: 'Connect ChatGPT to use GPT-5.5 on this app.',
    chatGptConsentTitle: 'Use your ChatGPT plan',
    chatGptConsent:
      'ShermanChat will send your prompts, manual context, and attached photos through this server to your ChatGPT account. Tokens stay in an HttpOnly cookie-backed session and signing out deletes the app session.',
    continueChatGpt: 'Continue',
    verificationCode: 'Verification code',
    openVerification: 'Open verification',
    checkingLogin: 'Checking login...',
  },
  he: {
    appName: 'ShermanAI',
    productName: 'ShermanChat',
    settings: 'הגדרות',
    newChat: 'שיחה חדשה',
    uiLanguage: 'ממשק',
    voiceLanguage: 'קול',
    english: 'English',
    hebrew: 'עברית',
    auto: 'אוטומטי',
    cell: 'תא',
    software: 'תוכנה',
    attachPhoto: 'צרף תמונה',
    startVoice: 'התחל תמלול קולי',
    stopVoice: 'עצור תמלול קולי',
    voiceUnavailable: 'תמלול קולי לא זמין בדפדפן הזה.',
    voiceError: 'התמלול הקולי נעצר לפני שנקלט טקסט.',
    placeholder: 'שאל את ShermanAI...',
    send: 'שלח',
    thinking: 'חושב...',
    sources: 'מקורות שנבדקו',
    notFound: 'אין מקור מאושר',
    needsDetail: 'צריך עוד פרט',
    chat: 'שיחה',
    model: 'מודל',
    mock: 'mock',
    removeAttachment: 'הסר קובץ',
    introTitle: 'איך אפשר לעזור בתא היום?',
    introSubtitle: 'שאלו טבעי. ShermanAI מחפש במדריכים רק כשצריך מקור מאושר.',
    photoReady: 'תמונה צורפה',
    connectChatGpt: 'חיבור ChatGPT',
    disconnectChatGpt: 'ניתוק',
    chatGptConnected: 'ChatGPT מחובר',
    chatGptRequired: 'חברו ChatGPT כדי להשתמש ב-GPT-5.5 באפליקציה.',
    chatGptConsentTitle: 'שימוש במנוי ChatGPT שלך',
    chatGptConsent:
      'ShermanChat ישלח את השאלות, הקשר מהמדריכים ותמונות מצורפות דרך השרת הזה לחשבון ChatGPT שלך. הטוקנים נשארים בסשן HttpOnly, וניתוק מוחק את סשן האפליקציה.',
    continueChatGpt: 'המשך',
    verificationCode: 'קוד אימות',
    openVerification: 'פתח אימות',
    checkingLogin: 'בודק התחברות...',
  },
} satisfies Record<UiLanguage, Record<string, string>>

const voiceLanguageLabels: Record<VoiceLanguage, Record<UiLanguage, string>> = {
  'en-US': { en: 'English', he: 'English' },
  'he-IL': { en: 'Hebrew', he: 'עברית' },
  'ru-RU': { en: 'Russian', he: 'Русский' },
}

const examples: Record<ChatMode, string[]> = {
  ask: [
    'How do I import a DXF file with bending lines?',
    'Who is allowed to open the electrical cabinet?',
    'What can you help me with?',
  ],
  troubleshoot: [
    'How do I fix a movement of an arm?',
    'BendMaster is not moving after switching on',
    'What should I check after an emergency stop?',
  ],
  procedure: [
    'How do I reference the Z, A, B, C, Z1 and Z2 axes?',
    'How should BendMaster be stopped after a malfunction?',
    'How do I define the upper machining side?',
  ],
  quality: [
    'What should I verify before continuing production?',
    'Which checks matter for bending settings in 3D design?',
    'What quality checks apply after a manual movement issue?',
  ],
  summarize: [
    'Summarize the ToolMaster purpose',
    'Summarize the DXF bending lines procedure',
    'Summarize emergency stop actions',
  ],
}

function formatMs(value?: number) {
  if (value === undefined) return ''
  if (value < 1000) return `${Math.round(value)} ms`
  return `${(value / 1000).toFixed(1)} s`
}

function profileForContext(context: ContextMode): ManualProfile {
  return context === 'auto' ? 'cell_operation' : context
}

function sourceUrl(item: ManualEvidence) {
  return item.page_image?.url || item.crop?.url || '#'
}

function responseLabel(response: ManualChatResponse, uiLanguage: UiLanguage) {
  const t = copy[uiLanguage]
  if (response.assistant_mode === 'chat') return t.chat
  if (response.citations.length > 0) return t.sources
  if (response.support_state === 'clarification') return t.needsDetail
  return t.notFound
}

function responseTone(response: ManualChatResponse) {
  if (response.assistant_mode === 'chat') return 'bg-slate-100 text-slate-700'
  if (response.citations.length > 0) return 'bg-emerald-50 text-emerald-700'
  if (response.support_state === 'not_found') return 'bg-amber-50 text-amber-800'
  return 'bg-sky-50 text-sky-700'
}

export default function ShermanChat() {
  const [mode, setMode] = useState<ChatMode>('ask')
  const [context, setContext] = useState<ContextMode>('auto')
  const [uiLanguage, setUiLanguage] = useState<UiLanguage>('en')
  const [voiceLanguage, setVoiceLanguage] = useState<VoiceLanguage>('en-US')
  const [showSettings, setShowSettings] = useState(false)
  const [message, setMessage] = useState('')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [attachments, setAttachments] = useState<ManualAttachmentResponse[]>([])
  const [isSending, setIsSending] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [llmProvider, setLlmProvider] = useState<string>('mock')
  const [chatGptSession, setChatGptSession] = useState<ChatGptSessionResponse>({ status: 'loading' })
  const [showChatGptConsent, setShowChatGptConsent] = useState(false)
  const [isConnectingChatGpt, setIsConnectingChatGpt] = useState(false)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [error, setError] = useState('')
  const [voiceError, setVoiceError] = useState('')
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const t = copy[uiLanguage]
  const requiresChatGpt = llmProvider === 'chatgpt_oauth'
  const isChatGptAuthenticated = chatGptSession.status === 'authenticated'

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [turns, isSending])

  useEffect(() => {
    let cancelled = false
    async function loadRuntimeStatus() {
      try {
        const [health, session] = await Promise.all([healthApi.check(), chatGptAuthApi.session()])
        if (cancelled) return
        setLlmProvider(String(health.provider || 'mock'))
        setChatGptSession(session)
        if (session.status === 'authenticated') {
          try {
            setAvailableModels(await chatGptAuthApi.models())
          } catch {
            setAvailableModels([])
          }
        }
      } catch {
        if (!cancelled) setChatGptSession({ status: 'unauthenticated' })
      }
    }
    void loadRuntimeStatus()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!requiresChatGpt || chatGptSession.status !== 'pending') return undefined
    const intervalMs = Math.max(2000, (chatGptSession.interval || 5) * 1000)
    const timer = window.setInterval(async () => {
      try {
        const next = await chatGptAuthApi.status()
        setChatGptSession(next)
        if (next.status === 'authenticated') {
          setIsConnectingChatGpt(false)
          setShowChatGptConsent(false)
          try {
            setAvailableModels(await chatGptAuthApi.models())
          } catch {
            setAvailableModels([])
          }
        }
      } catch (err: unknown) {
        setError(getErrorMessage(err))
      }
    }, intervalMs)
    return () => window.clearInterval(timer)
  }, [chatGptSession.interval, chatGptSession.status, requiresChatGpt])

  const resetChat = () => {
    setTurns([])
    setError('')
    setVoiceError('')
    setAttachments([])
    setMessage('')
  }

  const handlePhoto = async (file: File) => {
    setError('')
    try {
      const uploaded = await manualAssistantApi.uploadPhoto(file)
      setAttachments((current) => [...current, uploaded])
    } catch (err: unknown) {
      setError(getErrorMessage(err))
    }
  }

  const startChatGptLogin = async () => {
    setError('')
    setIsConnectingChatGpt(true)
    try {
      const pending = await chatGptAuthApi.login()
      setChatGptSession(pending)
      setShowChatGptConsent(false)
      if (pending.verificationUrl) {
        window.open(pending.verificationUrl, 'sherman-chatgpt-login', 'noopener,noreferrer,width=520,height=720')
      }
      if (pending.userCode && navigator.clipboard) {
        try {
          await navigator.clipboard.writeText(pending.userCode)
        } catch {
          // The code remains visible if clipboard access is blocked.
        }
      }
    } catch (err: unknown) {
      setError(getErrorMessage(err))
      setIsConnectingChatGpt(false)
    }
  }

  const disconnectChatGpt = async () => {
    setError('')
    try {
      const next = await chatGptAuthApi.logout()
      setChatGptSession(next)
      setAvailableModels([])
    } catch (err: unknown) {
      setError(getErrorMessage(err))
    }
  }

  const toggleVoice = () => {
    setVoiceError('')
    if (isListening) {
      recognitionRef.current?.stop()
      setIsListening(false)
      return
    }

    const speechWindow = window as SpeechWindow
    const Recognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition
    if (!Recognition) {
      setVoiceError(t.voiceUnavailable)
      return
    }

    const recognition = new Recognition()
    recognition.lang = voiceLanguage
    recognition.continuous = false
    recognition.interimResults = false
    recognition.onresult = (event) => {
      const transcripts: string[] = []
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index]
        const transcript = result?.[0]?.transcript?.trim()
        if (result?.isFinal && transcript) transcripts.push(transcript)
      }
      if (transcripts.length > 0) {
        setMessage((current) => `${current} ${transcripts.join(' ')}`.trim())
      }
    }
    recognition.onerror = () => {
      setVoiceError(t.voiceError)
      setIsListening(false)
    }
    recognition.onend = () => {
      setIsListening(false)
      recognitionRef.current = null
    }
    recognitionRef.current = recognition
    setIsListening(true)
    recognition.start()
  }

  const sendMessage = async (override?: string) => {
    const text = (override ?? message).trim()
    if (!text && attachments.length === 0) return
    if (requiresChatGpt && !isChatGptAuthenticated) {
      setError(t.chatGptRequired)
      setShowChatGptConsent(true)
      return
    }

    const startedAt = performance.now()
    const turnAttachments = attachments
    const userText = text || '[image attached]'
    setError('')
    setIsSending(true)
    setMessage('')
    setAttachments([])
    setTurns((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: 'user',
        text: userText,
        attachments: turnAttachments,
      },
    ])

    try {
      const response = await manualAssistantApi.shermanChat({
        profile: profileForContext(context),
        message: userText,
        attachment_ids: turnAttachments.map((item) => item.attachment_id),
        ui_language: uiLanguage,
        answer_language: 'follow_ui',
        retrieval_profile: 'accurate',
      })
      setTurns((current) => [
        ...current,
        {
          id: response.request_id,
          role: 'assistant',
          text: response.answer,
          response,
          durationMs: performance.now() - startedAt,
        },
      ])
    } catch (err: unknown) {
      setError(getErrorMessage(err))
    } finally {
      setIsSending(false)
    }
  }

  const activeExamples = examples[mode]

  return (
    <div dir={uiLanguage === 'he' ? 'rtl' : 'ltr'} className="h-[100dvh] overflow-hidden bg-[#f7f7f5] text-slate-950">
      <div className="flex h-full flex-col">
        <header className="relative z-20 flex min-h-14 items-center justify-between border-b border-black/5 bg-[#f7f7f5]/90 px-3 backdrop-blur-xl sm:px-5">
          <div className="inline-flex min-h-10 items-center gap-2 px-2 text-sm font-semibold text-slate-900 sm:px-3">
            <h1 className="inline-flex items-center">
            <img
              src={SHERMAN_LOGO_SRC}
              alt="Sherman Tailoring Integrated Solutions"
              className="h-6 w-auto max-w-[130px] object-contain sm:h-7 sm:max-w-[170px]"
            />
              <span className="sr-only">{t.appName}</span>
            </h1>
          </div>

          <div className="flex items-center gap-1.5">
            {requiresChatGpt && (
              <ChatGptStatusButton
                t={t}
                session={chatGptSession}
                isConnecting={isConnectingChatGpt}
                onConnect={() => setShowChatGptConsent(true)}
                onDisconnect={() => void disconnectChatGpt()}
              />
            )}
            <button
              type="button"
              onClick={resetChat}
              className="flex h-10 min-w-10 items-center justify-center gap-2 rounded-full border border-black/10 bg-white/70 px-2 text-sm font-medium text-slate-700 shadow-sm hover:bg-white sm:min-h-9 sm:w-auto sm:px-3"
              aria-label={t.newChat}
            >
              <Plus className="h-4 w-4" />
              <span className="hidden sm:inline">{t.newChat}</span>
            </button>
            <button
              type="button"
              onClick={() => setShowSettings((value) => !value)}
              className="flex h-10 w-10 items-center justify-center rounded-full border border-black/10 bg-white/70 text-slate-700 shadow-sm hover:bg-white"
              aria-label={t.settings}
            >
              <Settings className="h-4 w-4" />
            </button>
          </div>
        </header>

        {showSettings && (
          <section className="relative z-10 border-b border-black/5 bg-white/80 px-3 py-3 shadow-sm backdrop-blur-xl sm:px-5">
            <div className="mx-auto grid max-w-3xl gap-3 sm:grid-cols-2">
              <SettingGroup icon={Languages} label={t.uiLanguage}>
                {(['en', 'he'] as UiLanguage[]).map((language) => (
                  <SegmentButton key={language} active={uiLanguage === language} onClick={() => setUiLanguage(language)}>
                    {language === 'en' ? t.english : t.hebrew}
                  </SegmentButton>
                ))}
              </SettingGroup>
              <SettingGroup icon={Mic} label={t.voiceLanguage}>
                {(['en-US', 'he-IL', 'ru-RU'] as VoiceLanguage[]).map((language) => (
                  <SegmentButton
                    key={language}
                    active={voiceLanguage === language}
                    onClick={() => setVoiceLanguage(language)}
                  >
                    {voiceLanguageLabels[language][uiLanguage]}
                  </SegmentButton>
                ))}
              </SettingGroup>
            </div>
          </section>
        )}

        {requiresChatGpt && (showChatGptConsent || chatGptSession.status === 'pending') && (
          <ChatGptConnectPanel
            t={t}
            session={chatGptSession}
            isConnecting={isConnectingChatGpt}
            availableModels={availableModels}
            onContinue={() => void startChatGptLogin()}
            onCancel={() => setShowChatGptConsent(false)}
          />
        )}

        <main ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-3 pb-4 pt-4 sm:px-5">
          <div className="mx-auto flex min-h-full w-full max-w-3xl flex-col">
            {turns.length === 0 ? (
              <EmptyState
                activeExamples={activeExamples}
                mode={mode}
                setMode={setMode}
                uiLanguage={uiLanguage}
                onExample={(example) => void sendMessage(example)}
              />
            ) : (
              <div className="flex flex-1 flex-col gap-5 py-2">
                {turns.map((turn) => (
                  <ChatMessage
                    key={turn.id}
                    turn={turn}
                    uiLanguage={uiLanguage}
                    responseLabel={turn.response ? responseLabel(turn.response, uiLanguage) : undefined}
                  />
                ))}
                {isSending && (
                  <div className="flex items-center gap-3 px-1 text-sm text-slate-500">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-white shadow-sm">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </span>
                    {t.thinking}
                  </div>
                )}
              </div>
            )}
          </div>
        </main>

        <footer className="shrink-0 bg-gradient-to-t from-[#f7f7f5] via-[#f7f7f5] to-[#f7f7f5]/0 px-2 pb-[calc(env(safe-area-inset-bottom)+8px)] pt-2 sm:px-5 sm:pb-5">
          <div className="mx-auto max-w-3xl">
            {(error || voiceError) && (
              <div className="mb-2 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-900 shadow-sm">
                {error || voiceError}
              </div>
            )}
            {attachments.length > 0 && (
              <div className="mb-2 flex flex-wrap gap-2 px-1">
                {attachments.map((attachment) => (
                  <span
                    key={attachment.attachment_id}
                    className="inline-flex min-h-9 items-center gap-2 rounded-full border border-black/10 bg-white px-3 text-xs text-slate-700 shadow-sm"
                  >
                    <Paperclip className="h-3.5 w-3.5" />
                    {t.photoReady}
                    <button
                      type="button"
                      onClick={() =>
                        setAttachments((current) =>
                          current.filter((item) => item.attachment_id !== attachment.attachment_id)
                        )
                      }
                      className="flex h-6 w-6 items-center justify-center rounded-full hover:bg-black/5"
                      aria-label={`${t.removeAttachment} ${attachment.filename}`}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </span>
                ))}
              </div>
            )}

            <Composer
              t={t}
              context={context}
              setContext={setContext}
              message={message}
              setMessage={setMessage}
              isSending={isSending}
              isListening={isListening}
              attachments={attachments}
              fileInputRef={fileInputRef}
              handlePhoto={handlePhoto}
              toggleVoice={toggleVoice}
              sendMessage={() => void sendMessage()}
            />
          </div>
        </footer>
      </div>
    </div>
  )
}

function ChatGptStatusButton({
  t,
  session,
  isConnecting,
  onConnect,
  onDisconnect,
}: {
  t: Record<string, string>
  session: ChatGptSessionResponse
  isConnecting: boolean
  onConnect: () => void
  onDisconnect: () => void
}) {
  const connected = session.status === 'authenticated'
  return (
    <button
      type="button"
      onClick={connected ? onDisconnect : onConnect}
      className={clsx(
        'flex min-h-10 items-center justify-center gap-2 rounded-full border px-3 text-xs font-semibold shadow-sm transition',
        connected
          ? 'border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
          : 'border-black/10 bg-white/70 text-slate-700 hover:bg-white'
      )}
      aria-label={connected ? t.disconnectChatGpt : t.connectChatGpt}
    >
      {isConnecting || session.status === 'pending' || session.status === 'loading' ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : connected ? (
        <CheckCircle2 className="h-4 w-4" />
      ) : (
        <Bot className="h-4 w-4" />
      )}
      <span className="inline sm:hidden">GPT</span>
      <span className="hidden sm:inline">{connected ? t.chatGptConnected : t.connectChatGpt}</span>
    </button>
  )
}

function ChatGptConnectPanel({
  t,
  session,
  isConnecting,
  availableModels,
  onContinue,
  onCancel,
}: {
  t: Record<string, string>
  session: ChatGptSessionResponse
  isConnecting: boolean
  availableModels: string[]
  onContinue: () => void
  onCancel: () => void
}) {
  const pending = session.status === 'pending'
  return (
    <section className="relative z-10 border-b border-black/5 bg-white/85 px-3 py-3 shadow-sm backdrop-blur-xl sm:px-5">
      <div className="mx-auto flex max-w-3xl flex-col gap-3 rounded-[24px] border border-black/10 bg-white p-4 shadow-sm">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-950">{pending ? t.checkingLogin : t.chatGptConsentTitle}</h2>
            <p className="mt-1 text-sm leading-6 text-slate-600">{pending ? t.chatGptRequired : t.chatGptConsent}</p>
          </div>
          <button
            type="button"
            onClick={onCancel}
            className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full text-slate-500 hover:bg-slate-100"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {pending && session.userCode && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-500">
              {t.verificationCode}
            </span>
            <code className="rounded-full bg-slate-950 px-3 py-1.5 text-sm font-semibold tracking-wider text-white">
              {session.userCode}
            </code>
            {session.verificationUrl && (
              <a
                href={session.verificationUrl}
                target="_blank"
                rel="noreferrer"
                className="rounded-full border border-black/10 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                {t.openVerification}
              </a>
            )}
          </div>
        )}

        {availableModels.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {availableModels.slice(0, 5).map((model) => (
              <span key={model} className="rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-600">
                {model}
              </span>
            ))}
          </div>
        )}

        {!pending && (
          <div className="flex justify-end">
            <button
              type="button"
              onClick={onContinue}
              disabled={isConnecting}
              className="inline-flex min-h-10 items-center gap-2 rounded-full bg-slate-950 px-4 text-sm font-semibold text-white shadow-sm hover:bg-slate-800 disabled:bg-slate-300"
            >
              {isConnecting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Bot className="h-4 w-4" />}
              {t.continueChatGpt}
            </button>
          </div>
        )}
      </div>
    </section>
  )
}

function EmptyState({
  activeExamples,
  mode,
  setMode,
  uiLanguage,
  onExample,
}: {
  activeExamples: string[]
  mode: ChatMode
  setMode: (mode: ChatMode) => void
  uiLanguage: UiLanguage
  onExample: (example: string) => void
}) {
  const t = copy[uiLanguage]
  return (
    <section className="flex flex-1 flex-col justify-center pb-4">
      <div className="mx-auto mb-7 max-w-xl text-center">
        <div className="mx-auto mb-4 inline-flex min-h-16 items-center justify-center rounded-[24px] bg-white px-5 shadow-sm ring-1 ring-black/5">
          <img
            src={SHERMAN_LOGO_SRC}
            alt="Sherman Tailoring Integrated Solutions"
            className="h-9 w-auto max-w-[210px] object-contain"
          />
        </div>
        <h2 className="text-balance text-2xl font-semibold text-slate-950 sm:text-3xl">{t.introTitle}</h2>
        <p className="mt-2 text-sm leading-6 text-slate-500">{t.introSubtitle}</p>
      </div>

      <div className="mb-4 flex gap-2 overflow-x-auto pb-1">
        {(Object.keys(modeLabels.en) as ChatMode[]).map((item) => {
          const Icon = modeIcons[item]
          return (
            <button
              key={item}
              type="button"
              onClick={() => setMode(item)}
              className={clsx(
                'inline-flex min-h-10 min-w-max items-center gap-2 rounded-full border px-4 text-sm font-medium shadow-sm transition',
                mode === item
                  ? 'border-slate-950 bg-slate-950 text-white'
                  : 'border-black/10 bg-white text-slate-600 hover:bg-slate-50'
              )}
            >
              <Icon className="h-4 w-4" />
              {modeLabels[uiLanguage][item]}
            </button>
          )
        })}
      </div>

      <div className="grid gap-2 sm:grid-cols-3">
        {activeExamples.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onExample(example)}
            className="min-h-[76px] rounded-[22px] border border-black/10 bg-white px-4 py-3 text-left text-sm leading-5 text-slate-700 shadow-sm transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md"
          >
            {example}
          </button>
        ))}
      </div>
    </section>
  )
}

function ChatMessage({
  turn,
  uiLanguage,
  responseLabel,
}: {
  turn: ChatTurn
  uiLanguage: UiLanguage
  responseLabel?: string
}) {
  const isUser = turn.role === 'user'
  const t = copy[uiLanguage]

  return (
    <article className={clsx('flex gap-3', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div className="mt-1 hidden h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-black/5 sm:flex">
          <Bot className="h-4 w-4 text-slate-700" />
        </div>
      )}
      <div className={clsx('min-w-0', isUser ? 'max-w-[86%]' : 'max-w-full flex-1')}>
        <div
          className={clsx(
            isUser
              ? 'rounded-[24px] bg-slate-950 px-4 py-3 text-white shadow-sm'
              : 'rounded-[28px] bg-white px-4 py-4 text-slate-900 shadow-sm ring-1 ring-black/5 sm:px-5'
          )}
        >
          {turn.response && (
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <span
                className={clsx(
                  'inline-flex min-h-7 items-center rounded-full px-2.5 text-xs font-medium',
                  responseTone(turn.response)
                )}
              >
                {turn.response.citations.length > 0 ? (
                  <CheckCircle2 className="mr-1.5 h-3.5 w-3.5" />
                ) : turn.response.support_state === 'not_found' ? (
                  <AlertTriangle className="mr-1.5 h-3.5 w-3.5" />
                ) : (
                  <Bot className="mr-1.5 h-3.5 w-3.5" />
                )}
                {responseLabel}
              </span>
              <span className="inline-flex min-h-7 items-center rounded-full bg-slate-100 px-2.5 text-xs text-slate-600">
                {turn.response.model}
                {turn.response.provider ? ` · ${turn.response.provider}` : ''}
              </span>
              {turn.durationMs !== undefined && (
                <span className="inline-flex min-h-7 items-center rounded-full bg-slate-100 px-2.5 text-xs text-slate-600">
                  {formatMs(turn.durationMs)}
                </span>
              )}
            </div>
          )}

          <p className="whitespace-pre-wrap text-[15px] leading-7">{turn.text}</p>

          {turn.attachments && turn.attachments.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {turn.attachments.map((attachment) => (
                <span
                  key={attachment.attachment_id}
                  className="inline-flex min-h-8 items-center rounded-full bg-white/10 px-3 text-xs"
                >
                  <Paperclip className="mr-1.5 h-3.5 w-3.5" />
                  {attachment.filename}
                </span>
              ))}
            </div>
          )}

          {turn.response?.warnings?.map((warning) => (
            <div key={warning} className="mt-3 rounded-2xl bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800">
              {warning}
            </div>
          ))}

          {turn.response && turn.response.citations.length > 0 && (
            <Sources evidence={turn.response.citations} uiLanguage={uiLanguage} />
          )}
        </div>
        {!isUser && turn.response?.assistant_mode === 'manual_rag_tool' && (
          <p className="mt-2 px-2 text-xs text-slate-400">{t.sources}</p>
        )}
      </div>
    </article>
  )
}

function Sources({ evidence, uiLanguage }: { evidence: ManualEvidence[]; uiLanguage: UiLanguage }) {
  const t = copy[uiLanguage]
  return (
    <details className="mt-4 rounded-[22px] bg-slate-50 p-2 open:ring-1 open:ring-black/5">
      <summary className="flex min-h-10 cursor-pointer list-none items-center justify-between gap-3 rounded-full px-3 text-sm font-medium text-slate-700">
        <span className="inline-flex items-center gap-2">
          <FileText className="h-4 w-4 text-slate-500" />
          {t.sources}
          <span className="rounded-full bg-white px-2 py-0.5 text-xs text-slate-500">{evidence.length}</span>
        </span>
        <ChevronDown className="h-4 w-4 text-slate-400" />
      </summary>
      <div className="grid gap-2 pt-2">
        {evidence.map((item, index) => (
          <a
            key={`${item.citation_id}-${index}`}
            href={sourceUrl(item)}
            target="_blank"
            rel="noreferrer"
            className="rounded-[18px] bg-white p-3 text-left shadow-sm ring-1 ring-black/5 transition hover:bg-slate-50"
          >
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="min-w-0 truncate text-xs font-semibold text-slate-800">
                [{index + 1}] {item.manual_id}
              </span>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                p.{item.page_number}
              </span>
            </div>
            <p className="line-clamp-3 text-xs leading-5 text-slate-600">{item.excerpt}</p>
          </a>
        ))}
      </div>
    </details>
  )
}

function Composer({
  t,
  context,
  setContext,
  message,
  setMessage,
  isSending,
  isListening,
  attachments,
  fileInputRef,
  handlePhoto,
  toggleVoice,
  sendMessage,
}: {
  t: Record<string, string>
  context: ContextMode
  setContext: (context: ContextMode) => void
  message: string
  setMessage: (message: string) => void
  isSending: boolean
  isListening: boolean
  attachments: ManualAttachmentResponse[]
  fileInputRef: React.RefObject<HTMLInputElement>
  handlePhoto: (file: File) => Promise<void>
  toggleVoice: () => void
  sendMessage: () => void
}) {
  return (
    <div className="rounded-[30px] border border-black/10 bg-white p-2 shadow-[0_18px_48px_rgba(15,23,42,0.16)]">
      <div className="mb-1 flex gap-1 overflow-x-auto px-1 pt-1">
        {(['auto', 'cell_operation', 'software'] as ContextMode[]).map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => setContext(item)}
            className={clsx(
              'min-h-8 min-w-max rounded-full px-3 text-xs font-semibold transition',
              context === item ? 'bg-slate-950 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            )}
          >
            {item === 'auto' ? t.auto : item === 'cell_operation' ? t.cell : t.software}
          </button>
        ))}
      </div>

      <div className="flex items-end gap-1.5">
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
        <RoundIconButton label={t.attachPhoto} onClick={() => fileInputRef.current?.click()}>
          <Camera className="h-5 w-5" />
        </RoundIconButton>
        <RoundIconButton label={isListening ? t.stopVoice : t.startVoice} onClick={toggleVoice} active={isListening}>
          {isListening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
        </RoundIconButton>
        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault()
              sendMessage()
            }
          }}
          rows={1}
          placeholder={t.placeholder}
          className="max-h-32 min-h-12 flex-1 resize-none bg-transparent px-2 py-3 text-[15px] leading-6 text-slate-900 outline-none placeholder:text-slate-400"
        />
        <button
          type="button"
          onClick={sendMessage}
          disabled={isSending || (!message.trim() && attachments.length === 0)}
          className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-slate-950 text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          aria-label={t.send}
        >
          {isSending ? <Loader2 className="h-5 w-5 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
        </button>
      </div>
    </div>
  )
}

function RoundIconButton({
  label,
  onClick,
  active = false,
  children,
}: {
  label: string
  onClick: () => void
  active?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full transition',
        active ? 'bg-sky-50 text-sky-700' : 'text-slate-600 hover:bg-slate-100'
      )}
      aria-label={label}
    >
      {children}
    </button>
  )
}

function SettingGroup({
  icon: Icon,
  label,
  children,
}: {
  icon: typeof Languages
  label: string
  children: React.ReactNode
}) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase text-slate-500">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <div className="flex gap-2 overflow-x-auto">{children}</div>
    </div>
  )
}

function SegmentButton({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'min-h-9 min-w-max rounded-full px-3 text-sm font-medium transition',
        active ? 'bg-slate-950 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
      )}
    >
      {children}
    </button>
  )
}
