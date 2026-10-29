/** Steps 2 and 3: folded CAD (STEP) and the PDF drawing, through the existing catalog upload routes. */
import { useState } from 'react'
import { onboardingApi } from '../api/onboardingApi'
import { fill } from '../copy'
import { isStepCad, mergedPart } from '../model/checklist'
import type { ProductionApiError } from '../model/types'
import { ApiErrorNotice, FileCard, FileDrop, Notice, ReasonRow, displayFileName, fillNodes, formatBytes, rememberFileName, shortHash } from '../components/common'
import StaleDraftFix from '../components/StaleDraftFix'
import type { StepProps } from './stepProps'

function useUpload() {
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [error, setError] = useState<ProductionApiError | null>(null)
  const [message, setMessage] = useState<{ tone: 'info' | 'attention' | 'blocked'; text: string } | null>(null)
  const [sent, setSent] = useState<File | null>(null)
  return { busy, setBusy, progress, setProgress, error, setError, message, setMessage, sent, setSent }
}

export function CadStep({ partId, snapshot, summary, t, language, refresh }: StepProps) {
  const part = mergedPart(snapshot)
  const u = useUpload()
  const upload = async (file: File) => {
    u.setBusy(true); u.setError(null); u.setMessage(null); u.setProgress(0); u.setSent(file)
    try {
      const result = await onboardingApi.uploadCad(partId, file, u.setProgress)
      if (result.status === 'CAD_PROCESSED') u.setMessage({ tone: 'info', text: fill(t.cadProcessed, { n: result.bends_detected ?? 0 }) })
      else if (result.status === 'CAD_UPLOADED_UNANALYZED') u.setMessage({ tone: 'attention', text: t.cadUnanalysed })
      else if (result.status === 'CAD_UPLOAD_SUPERSEDED_NOT_PUBLISHED') u.setMessage({ tone: 'attention', text: t.cadSuperseded })
      else u.setMessage({ tone: 'blocked', text: `${t.cadAnalysisFailed}${result.bend_analysis_error ? ` (${result.bend_analysis_error})` : ''}` })
      await rememberFileName(file)
      // Done uploading: drop the progress card before the (slower) re-read, so it never lingers under the new file.
      u.setBusy(false); u.setProgress(null)
      await refresh()
    } catch (error) {
      u.setError(error as ProductionApiError)
    } finally {
      u.setBusy(false); u.setProgress(null)
    }
  }
  const hasCad = !!part?.cad_file_hash
  const cadName = displayFileName(part?.cad_file_hash, part?.cad_file_path || snapshot.vendorSetup?.part.cad_file_path, t.cadFileFallback)
  const sentHere = u.sent && u.sent.name === cadName ? u.sent : null
  const n = summary.candidateCount
  return (
    <div className="onb-stack">
      {hasCad && (
        <FileCard data-cad-current="" name={cadName || 'CAD'} kind={isStepCad(part) ? 'STEP' : undefined}
          facts={<>{sentHere ? <span className="onb-num">{formatBytes(sentHere.size)} · </span> : null}{t.fileHash} <bdi dir="ltr" className="onb-num">{shortHash(part?.cad_file_hash)}</bdi></>}
          result={snapshot.vendorSetup ? (n > 0 ? <span>{fillNodes(t.cadResult, { n: <bdi className="onb-num" data-candidate-count>{n}</bdi> })}</span>
            : <span>{t.cadResultNone}<bdi className="sr-only" data-candidate-count>{n}</bdi></span>) : undefined}
          resultTone={n > 0 ? 'done' : 'attention'} />
      )}
      {hasCad && !isStepCad(part) && <ReasonRow code="CAD_NOT_STEP" language={language} />}
      {hasCad && isStepCad(part) && snapshot.vendorSetup && summary.candidateCount === 0 && summary.activeCount === 0
        && <ReasonRow code="CAD_NO_CANDIDATES" language={language} severity="WARNING" />}
      <FileDrop accept=".step,.stp" busy={u.busy} progress={u.progress} label={hasCad ? t.replaceFile : t.chooseFile} compact={hasCad}
        title={t.dropTitleCad} hint={t.cadAccept} onFile={file => void upload(file)} t={t} />
      {u.message && <Notice tone={u.message.tone} role="status"><span data-cad-result>{u.message.text}</span></Notice>}
      <ApiErrorNotice error={u.error} language={language} />
    </div>
  )
}

export function DrawingStep(props: StepProps) {
  const { partId, snapshot, t, language, refresh } = props
  const part = mergedPart(snapshot)
  const u = useUpload()
  const upload = async (file: File) => {
    u.setBusy(true); u.setError(null); u.setMessage(null); u.setProgress(0); u.setSent(file)
    try {
      await onboardingApi.uploadDrawing(partId, file, u.setProgress)
      u.setMessage({ tone: 'info', text: t.drawingAttached })
      await rememberFileName(file)
      u.setBusy(false); u.setProgress(null)
      await refresh()
    } catch (error) {
      u.setError(error as ProductionApiError)
    } finally {
      u.setBusy(false); u.setProgress(null)
    }
  }
  const has = !!part?.drawing_file_hash
  const name = displayFileName(part?.drawing_file_hash, part?.drawing_file_path || snapshot.vendorSetup?.part.drawing_file_path, t.drawingFileFallback)
  const sentHere = u.sent && u.sent.name === name ? u.sent : null
  return (
    <div className="onb-stack">
      {has && (
        <FileCard data-drawing-current="" name={name} kind="PDF"
          facts={<>{sentHere ? <span className="onb-num">{formatBytes(sentHere.size)} · </span> : null}{t.fileHash} <bdi dir="ltr" className="onb-num">{shortHash(part?.drawing_file_hash)}</bdi></>}
          result={t.drawingResult} />
      )}
      {has && <StaleDraftFix {...props} />}
      <FileDrop accept=".pdf,application/pdf" busy={u.busy} progress={u.progress} label={has ? t.replaceFile : t.chooseFile} compact={has}
        title={t.dropTitleDrawing} hint={t.drawingAccept} onFile={file => void upload(file)} t={t} />
      {u.message && <Notice tone={u.message.tone} role="status">{u.message.text}</Notice>}
      <ApiErrorNotice error={u.error} language={language} />
    </div>
  )
}
