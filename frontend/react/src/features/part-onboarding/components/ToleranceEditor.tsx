/**
 * Step 6: tolerances. Part defaults, per-bend overrides in sequence order
 * (including flange sides REQUIRED / NOT_INSPECTED and the included angle of
 * manual marks), general-measurement overrides, the inspection-stage choice and
 * the recipe thickness. Saved with PUT release-draft under an optimistic
 * revision; a 409 asks for a reload instead of overwriting someone else's save.
 */
import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Save } from 'lucide-react'
import { onboardingApi, onboardingKeys } from '../api/onboardingApi'
import { toleranceSourceCopy, type OnbCopy } from '../copy'
import { buildReleaseRequest } from '../model/checklist'
import { formFromRequest, requestFromForm, type BandText, type ToleranceForm } from '../model/toleranceForm'
import { TOLERANCE_SOURCES, type FlangeSideRequirement, type ProductionApiError } from '../model/types'
import { ApiErrorNotice, Notice } from './common'
import type { StepProps } from '../steps/stepProps'

function problemText(t: OnbCopy, codes: string[] | undefined): string | null {
  if (!codes?.length) return null
  return codes.map(code => (t as Record<string, string>)[`tolProblem${code}`] ?? code).join(' ')
}

function BandInput({ id, label, band, onChange, errors, t, unit, optional }: {
  id: string; label: string; band: BandText; onChange: (next: BandText) => void; errors?: string[]; t: OnbCopy; unit: string; optional?: boolean
}) {
  const invalid = !!errors?.length
  const message = problemText(t, errors)
  return (
    <fieldset className="grid gap-1.5" data-band={id}>
      <legend className="text-sm font-medium" style={{ color: 'var(--onb-steel)' }}>{label}{optional ? <span className="onb-help"> ({t.optionalBand})</span> : null}</legend>
      <div className="flex flex-wrap items-end gap-2">
        {band.symmetric ? (
          <label className="grid gap-0.5 text-xs onb-muted">±{unit}
            <input id={`${id}-plus`} className="onb-input onb-num w-28" inputMode="decimal" dir="ltr" value={band.plus} aria-invalid={invalid}
              onChange={event => onChange({ ...band, plus: event.target.value })} />
          </label>
        ) : <>
          <label className="grid gap-0.5 text-xs onb-muted">{t.plus} +{unit}
            <input id={`${id}-plus`} className="onb-input onb-num w-24" inputMode="decimal" dir="ltr" value={band.plus} aria-invalid={invalid}
              onChange={event => onChange({ ...band, plus: event.target.value })} />
          </label>
          <label className="grid gap-0.5 text-xs onb-muted">{t.minus} −{unit}
            <input id={`${id}-minus`} className="onb-input onb-num w-24" inputMode="decimal" dir="ltr" value={band.minus} aria-invalid={invalid}
              onChange={event => onChange({ ...band, minus: event.target.value })} />
          </label>
        </>}
        <label className="inline-flex min-h-11 items-center gap-2 text-sm">
          <input type="checkbox" checked={band.symmetric} onChange={event => onChange({ ...band, symmetric: event.target.checked, minus: event.target.checked ? band.plus : band.minus })} />
          {t.symmetric}
        </label>
      </div>
      {message && <span className="onb-error-text" data-band-error={id}>{message}</span>}
    </fieldset>
  )
}

function SideSelect({ value, onChange, label, t }: { value: FlangeSideRequirement; onChange: (v: FlangeSideRequirement) => void; label: string; t: OnbCopy }) {
  return (
    <select className="onb-select" aria-label={label} value={value} onChange={event => onChange(event.target.value as FlangeSideRequirement)}>
      <option value="REQUIRED">{t.required}</option>
      <option value="NOT_INSPECTED">{t.notInspected}</option>
    </select>
  )
}

function Editor({ partId, snapshot, summary, t, language, refresh, onSaved }: StepProps & { onSaved: (revision: number) => void }) {
  const queryClient = useQueryClient()
  const releaseDraft = snapshot.releaseDraft
  const [form, setForm] = useState<ToleranceForm>(() => formFromRequest(releaseDraft?.request, summary))
  const [showErrors, setShowErrors] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ProductionApiError | null>(null)
  const result = requestFromForm(form, summary, releaseDraft?.revision ?? null)
  const errors = showErrors ? result.errors : {}
  const set = <K extends keyof ToleranceForm>(key: K, value: ToleranceForm[K]) => setForm(previous => ({ ...previous, [key]: value }))
  const setBend = (fid: string, patch: Partial<ToleranceForm['bends'][string]>) => {
    setForm(previous => ({ ...previous, bends: { ...previous.bends, [fid]: { ...previous.bends[fid], ...patch } } }))
  }
  const conflict = error?.codes.includes('RELEASE_DRAFT_REVISION_CHANGED') || (error?.status === 409 && !error.codes.length)

  const save = async () => {
    setShowErrors(true); setError(null)
    if (Object.keys(result.errors).length) return
    setBusy(true)
    try {
      const body = buildReleaseRequest(result.request, summary, releaseDraft?.revision ?? null)
      const response = await onboardingApi.putReleaseDraft(partId, body)
      queryClient.setQueryData(onboardingKeys.releaseDraft(partId), response)
      onSaved(response.revision)
      await refresh()
    } catch (err) {
      setError(err as ProductionApiError)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="grid gap-7" onSubmit={event => { event.preventDefault(); void save() }} data-tolerance-editor>
      <p className="onb-lead">{t.tolerancesLead}</p>

      <section className="grid gap-4">
        <h3 className="onb-h3">{t.defaults}</h3>
        <div className="onb-grid-2">
          <BandInput id="angle" label={t.angleTol} unit="°" band={form.angle} onChange={v => set('angle', v)} errors={errors.angle} t={t} />
          <BandInput id="flange" label={t.flangeTol} unit="" band={form.flange} onChange={v => set('flange', v)} errors={errors.flange} t={t} />
          <BandInput id="bendLine" label={t.bendLineTol} unit="" band={form.bendLine} onChange={v => set('bendLine', v)} errors={errors.bendLine} t={t} optional />
          <BandInput id="backgauge" label={t.backgaugeTol} unit="" band={form.backgauge} onChange={v => set('backgauge', v)} errors={errors.backgauge} t={t} optional />
          <BandInput id="general" label={t.generalTol} unit="" band={form.general} onChange={v => set('general', v)} errors={errors.general} t={t} optional />
          <label className="onb-field"><span>{t.surfaceBand}</span>
            <input className="onb-input onb-num w-28" inputMode="decimal" dir="ltr" value={form.surface} aria-invalid={!!errors.surface}
              onChange={event => set('surface', event.target.value)} />
            <span className="onb-help" data-surface-default>{t.surfaceBandHint}</span>
            {errors.surface && <span className="onb-error-text">{problemText(t, errors.surface)}</span>}
          </label>
          <label className="onb-field"><span>{t.tolSource}</span>
            <select className="onb-select" value={form.source} aria-invalid={!!errors.source} data-tol-source
              onChange={event => set('source', event.target.value as ToleranceForm['source'])}>
              <option value="">—</option>
              {TOLERANCE_SOURCES.map(source => <option key={source} value={source}>{toleranceSourceCopy[language][source]}</option>)}
            </select>
            {errors.source && <span className="onb-error-text">{problemText(t, errors.source)}</span>}
          </label>
          <label className="onb-field"><span>{t.tolNote}</span>
            <input className="onb-input" maxLength={500} value={form.note} onChange={event => set('note', event.target.value)} />
          </label>
        </div>
      </section>

      <section className="grid gap-3">
        <h3 className="onb-h3">{t.inspectionStage}</h3>
        <div className="onb-choice-grid" role="radiogroup" aria-label={t.inspectionStage}>
          {(['FINAL', 'IN_PROCESS'] as const).map(stage => (
            <label key={stage} className="onb-note cursor-pointer" data-tone={form.stage === stage ? 'info' : 'neutral'}>
              <input type="radio" name="inspection-stage" value={stage} checked={form.stage === stage} onChange={() => set('stage', stage)} data-stage={stage} />
              <span className="grid gap-0.5"><strong>{t[`stage${stage}` as const]}</strong>
                <span className="onb-help">{stage === 'FINAL' ? t.stageFinalHint : t.stageInProcessHint}</span></span>
            </label>
          ))}
        </div>
      </section>

      {summary.ordered.length > 0 && <section className="grid gap-3">
        <h3 className="onb-h3">{t.overrides}</h3>
        <div className="onb-table-wrap">
          <table className="onb-table" data-override-table>
            <thead><tr><th>{t.seq}</th><th>{t.bendCol}</th><th>{t.angleTol}</th><th>{t.sideA}</th><th>{t.sideB}</th></tr></thead>
            <tbody>
              {summary.ordered.map(row => {
                const f = form.bends[row.feature_id]
                if (!f) return null
                const base = `bends.${row.feature_id}`
                return (
                  <tr key={row.feature_id} data-override-row={row.feature_id}>
                    <td className="onb-num">{row.step}{row.moveIndex ? <span className="onb-sub">{t.move} {row.moveIndex}</span> : null}</td>
                    <td><bdi dir="ltr" className="font-semibold">{row.label}</bdi>
                      {row.source === 'manual' && <label className="mt-2 grid gap-1 text-xs onb-muted">{t.manualAngle}
                        <input className="onb-input onb-num w-28" inputMode="decimal" dir="ltr" value={f.manualAngle} data-manual-angle={row.feature_id}
                          aria-invalid={!!errors[`${base}.manualAngle`]} onChange={event => setBend(row.feature_id, { manualAngle: event.target.value })} />
                        {errors[`${base}.manualAngle`] && <span className="onb-error-text">{t.manualAngleMissing}</span>}
                      </label>}
                    </td>
                    <td><BandInput id={`${base}.angle`} label={t.inherit} unit="°" band={f.angle} onChange={v => setBend(row.feature_id, { angle: v })} errors={errors[`${base}.angle`]} t={t} optional /></td>
                    <td className="grid gap-2"><SideSelect label={`${t.sideA} ${row.label}`} value={f.sideA} onChange={v => setBend(row.feature_id, { sideA: v })} t={t} />
                      {f.sideA === 'REQUIRED' && <BandInput id={`${base}.flangeA`} label={t.inherit} unit="" band={f.flangeA} onChange={v => setBend(row.feature_id, { flangeA: v })} errors={errors[`${base}.flangeA`]} t={t} optional />}</td>
                    <td className="grid gap-2"><SideSelect label={`${t.sideB} ${row.label}`} value={f.sideB} onChange={v => setBend(row.feature_id, { sideB: v })} t={t} />
                      {f.sideB === 'REQUIRED' && <BandInput id={`${base}.flangeB`} label={t.inherit} unit="" band={f.flangeB} onChange={v => setBend(row.feature_id, { flangeB: v })} errors={errors[`${base}.flangeB`]} t={t} optional />}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>}

      {summary.measurementIds.length > 0 && <section className="grid gap-3">
        <h3 className="onb-h3">{t.measurementsTitle}</h3>
        {summary.measurementIds.map(id => {
          const f = form.measurements[id]
          const label = snapshot.vendorSetup?.draft?.general_measurements?.find(item => item.id === id)?.label ?? id
          if (!f) return null
          return (
            <div key={id} className="flex flex-wrap items-end gap-4" data-measurement-row={id}>
              <BandInput id={`measurements.${id}`} label={label} unit="" band={f.tol} t={t} optional errors={errors[`measurements.${id}`]}
                onChange={v => setForm(previous => ({ ...previous, measurements: { ...previous.measurements, [id]: { ...previous.measurements[id], tol: v } } }))} />
              <label className="inline-flex min-h-11 items-center gap-2 text-sm">
                <input type="checkbox" checked={f.required !== false}
                  onChange={event => setForm(previous => ({ ...previous, measurements: { ...previous.measurements, [id]: { ...previous.measurements[id], required: event.target.checked } } }))} />
                {t.measurementRequired}
              </label>
            </div>
          )
        })}
      </section>}

      <section className="onb-grid-2">
        <label className="onb-field"><span>{t.thicknessOverride} <span className="onb-help">({t.optionalField})</span></span>
          <input className="onb-input onb-num w-32" inputMode="decimal" dir="ltr" value={form.thickness} aria-invalid={!!errors.thickness}
            onChange={event => set('thickness', event.target.value)} />
          {errors.thickness && <span className="onb-error-text">{t.thicknessInvalid}</span>}
        </label>
        <label className="onb-field"><span>{t.releaseNote}</span>
          <textarea className="onb-textarea" rows={2} maxLength={2000} value={form.releaseNote} onChange={event => set('releaseNote', event.target.value)} />
        </label>
      </section>

      <div className="onb-sticky-actions">
        <button type="submit" className="onb-btn onb-btn-primary" disabled={busy} data-save-tolerances><Save size={18} aria-hidden />{busy ? t.savingTolerances : t.saveTolerances}</button>
      </div>
      {conflict ? <Notice tone="attention" role="alert"><span className="grid gap-2"><span>{t.tolConflict}</span>
        <button type="button" className="onb-btn onb-btn-quiet justify-self-start" onClick={() => void refresh()}>{t.reload}</button></span></Notice>
        : <ApiErrorNotice error={error} language={language} />}
    </form>
  )
}

/** Remount the editor whenever the persisted draft or the bend setup revision changes. */
export default function ToleranceEditor(props: StepProps) {
  // The confirmation lives outside the keyed editor: our own save bumps the revision and remounts it.
  const [savedRevision, setSavedRevision] = useState<number | null>(null)
  const revision = props.snapshot.releaseDraft?.revision ?? 0
  const key = `${revision}:${props.summary.draftRevision}:${props.summary.ordered.map(row => row.feature_id).join(',')}`
  return <>
    <Editor key={key} {...props} onSaved={setSavedRevision} />
    {savedRevision != null && savedRevision === revision && <p className="mt-3" role="status" data-tolerances-saved>{props.t.tolerancesSaved}</p>}
  </>
}
