/**
 * /production/parts/new - create a production part. The canonical SKU preview
 * uses the same rule as the server (CONTRACTS §1); 409 SKU_COLLISION and
 * 422 SKU_INVALID come back as plain text.
 *
 * The catalog part has no thickness field, so a thickness typed here is saved as
 * the recipe thickness in the part's release draft (ReleaseRequestV1.thickness_mm,
 * an incomplete draft with no tolerances yet). If that save fails the part still
 * exists and the operator is told to set it in the Tolerances step.
 */
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import { ArrowLeft } from 'lucide-react'
import { onboardingApi, onboardingKeys } from '../api/onboardingApi'
import { onbCopy } from '../copy'
import { canonicalSkuPreview, emptyReleaseRequest } from '../model/checklist'
import type { ProductionApiError } from '../model/types'
import { ApiErrorNotice, LanguageToggle, Notice, useOnboardingLanguage } from '../components/common'
import '../onboarding.css'

export default function NewProductionPartPage() {
  const [language, setLanguage] = useOnboardingLanguage()
  const t = onbCopy[language]
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [fields, setFields] = useState({ part_number: '', part_name: '', revision: '', material: '', thickness: '', customer: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ProductionApiError | null>(null)
  const [createdId, setCreatedId] = useState<string | null>(null)
  const [touched, setTouched] = useState(false)
  const sku = canonicalSkuPreview(fields.part_number)
  const thicknessText = fields.thickness.trim().replace(',', '.')
  const thickness = thicknessText ? Number(thicknessText) : null
  const thicknessInvalid = thickness != null && !(Number.isFinite(thickness) && thickness > 0)
  const valid = sku.ok && !!fields.part_name.trim() && !thicknessInvalid
  const set = (key: keyof typeof fields) => (event: { target: { value: string } }) => { setError(null); setFields(previous => ({ ...previous, [key]: event.target.value })) }

  const submit = async () => {
    setTouched(true)
    if (!valid) return
    setBusy(true); setError(null)
    try {
      const body: Parameters<typeof onboardingApi.createPart>[0] = { part_number: fields.part_number.trim(), part_name: fields.part_name.trim() }
      if (fields.revision.trim()) body.revision = fields.revision.trim()
      if (fields.material.trim()) body.material = fields.material.trim()
      if (fields.customer.trim()) body.customer = fields.customer.trim()
      const created = await onboardingApi.createPart(body)
      await queryClient.invalidateQueries({ queryKey: onboardingKeys.parts })
      if (thickness != null) {
        try {
          await onboardingApi.putReleaseDraft(created.part.id, { ...emptyReleaseRequest(0, 0), thickness_mm: thickness })
        } catch (err) {
          setCreatedId(created.part.id)
          setError(err as ProductionApiError)
          return
        }
      }
      navigate(`/production/parts/${encodeURIComponent(created.part.id)}/onboard?step=cad`)
    } catch (err) {
      setError(err as ProductionApiError)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="onb-root" dir={language === 'he' ? 'rtl' : 'ltr'} lang={language} data-new-production-part>
      <header className="onb-header">
        <div className="grid gap-2">
          <Link to="/production/parts" className="onb-back"><ArrowLeft size={16} className={language === 'he' ? 'rotate-180' : ''} aria-hidden />{t.backToParts}</Link>
          <h1 className="text-2xl font-bold">{t.newPartTitle}</h1>
          <p className="onb-lead">{t.newPartLead}</p>
        </div>
        <LanguageToggle language={language} onChange={setLanguage} />
      </header>
      <form className="onb-panel onb-section grid max-w-3xl gap-5" onSubmit={event => { event.preventDefault(); void submit() }}>
        <div className="onb-grid-2">
          <label className="onb-field"><span>{t.partNumber}</span>
            <input className="onb-input" dir="ltr" autoComplete="off" value={fields.part_number} onChange={set('part_number')} aria-invalid={touched && !sku.ok} data-field="part_number" />
          </label>
          <div className="onb-field" aria-live="polite">
            <span>{t.skuPreview}</span>
            <div className="onb-tag" data-sku-preview={sku.ok ? sku.sku : ''}>
              <bdi dir="ltr" className="onb-tag-sku">{fields.part_number.trim() ? (sku.ok ? sku.sku : sku.sku || '—') : '—'}</bdi>
            </div>
            {!fields.part_number.trim() ? <span className="onb-help">{t.skuPreviewEmpty}</span>
              : !sku.ok && <span className="onb-error-text" data-sku-error={sku.reason}>{t[`skuInvalid${sku.reason}` as const]}</span>}
          </div>
          <label className="onb-field"><span>{t.partName}</span>
            <input className="onb-input" value={fields.part_name} onChange={set('part_name')} aria-invalid={touched && !fields.part_name.trim()} data-field="part_name" />
          </label>
          <label className="onb-field"><span>{t.revision} <span className="onb-help">({t.optionalField})</span></span>
            <input className="onb-input" dir="ltr" value={fields.revision} onChange={set('revision')} maxLength={32} data-field="revision" />
          </label>
          <label className="onb-field"><span>{t.material} <span className="onb-help">({t.optionalField})</span></span>
            <input className="onb-input" value={fields.material} onChange={set('material')} maxLength={128} data-field="material" />
          </label>
          <label className="onb-field"><span>{t.thickness} <span className="onb-help">({t.optionalField})</span></span>
            <input className="onb-input onb-num" dir="ltr" inputMode="decimal" value={fields.thickness} onChange={set('thickness')} aria-invalid={thicknessInvalid} data-field="thickness" />
            {thicknessInvalid && <span className="onb-error-text">{t.thicknessInvalid}</span>}
          </label>
          <label className="onb-field"><span>{t.customer} <span className="onb-help">({t.optionalField})</span></span>
            <input className="onb-input" value={fields.customer} onChange={set('customer')} data-field="customer" />
          </label>
        </div>
        {createdId && <Notice tone="attention"><span className="grid gap-2" data-thickness-not-saved>{t.thicknessNotSaved}
          <Link className="onb-btn onb-btn-primary justify-self-start" to={`/production/parts/${encodeURIComponent(createdId)}/onboard?step=tolerances`}>{t.continueOnboarding}</Link></span></Notice>}
        <ApiErrorNotice error={error} language={language} />
        {!createdId && <div className="flex flex-wrap gap-3">
          <button type="submit" className="onb-btn onb-btn-primary" disabled={busy || (touched && !valid)} data-create-part>{busy ? t.creating : t.createPart}</button>
          <Link to="/production/parts" className="onb-btn onb-btn-quiet">{t.cancel}</Link>
        </div>}
      </form>
    </div>
  )
}
