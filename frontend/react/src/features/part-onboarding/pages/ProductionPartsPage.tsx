/** /production/parts - production parts with their onboarding state and active recipe. */
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Plus } from 'lucide-react'
import { onboardingApi, onboardingKeys } from '../api/onboardingApi'
import { onbCopy, onboardingStateCopy } from '../copy'
import type { OnboardingState, ProductionApiError } from '../model/types'
import { ApiErrorNotice, LanguageToggle, useOnboardingLanguage } from '../components/common'
import '../onboarding.css'

function stateTone(state: OnboardingState): 'done' | 'attention' | undefined {
  return state === 'RELEASED' ? 'done' : state === 'RELEASED_STALE' ? 'attention' : undefined
}

function formatDate(value: string | null | undefined, language: 'en' | 'he'): string {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(language === 'he' ? 'he-IL' : 'en-GB', { dateStyle: 'medium', timeStyle: 'short' })
}

export default function ProductionPartsPage() {
  const [language, setLanguage] = useOnboardingLanguage()
  const t = onbCopy[language]
  const parts = useQuery({ queryKey: onboardingKeys.parts, queryFn: onboardingApi.listParts, retry: 1 })
  return (
    <div className="onb-root" dir={language === 'he' ? 'rtl' : 'ltr'} lang={language} data-production-parts>
      <header className="onb-header">
        <div className="grid gap-1">
          <h1 className="text-2xl font-bold">{t.productionParts}</h1>
          <p className="onb-lead">{t.partsLead}</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <LanguageToggle language={language} onChange={setLanguage} />
          <Link to="/production/parts/new" className="onb-btn onb-btn-primary" data-new-part><Plus size={18} aria-hidden />{t.newPart}</Link>
        </div>
      </header>
      {parts.isLoading && <p className="onb-muted">{t.loadingParts}</p>}
      {parts.isError && <div className="grid gap-3"><ApiErrorNotice error={parts.error as unknown as ProductionApiError} language={language} />
        <button type="button" className="onb-btn onb-btn-quiet justify-self-start" onClick={() => void parts.refetch()}>{t.retry}</button></div>}
      {parts.data && parts.data.length === 0 && <div className="onb-panel onb-section"><p>{t.noParts}</p></div>}
      {parts.data && parts.data.length > 0 && <div className="onb-table-wrap onb-panel">
        <table className="onb-table" data-parts-table>
          <thead><tr><th>{t.sku}</th><th>{t.name}</th><th>{t.revision}</th><th>{t.state}</th><th>{t.activeRecipe}</th><th><span className="sr-only">{t.onboard}</span></th></tr></thead>
          <tbody>
            {parts.data.map(part => (
              <tr key={part.part_id} data-part-row={part.part_id}>
                <td><bdi dir="ltr" className="font-semibold">{part.sku_display || part.sku || part.part_number}</bdi></td>
                <td>{part.part_name}</td>
                <td><bdi dir="ltr">{part.revision || '—'}</bdi></td>
                <td><span className="onb-chip" data-tone={stateTone(part.onboarding_state)} data-state={part.onboarding_state}>
                  {onboardingStateCopy[language][part.onboarding_state] ?? part.onboarding_state}</span></td>
                <td>{part.active_recipe ? <span className="grid"><bdi dir="ltr" className="font-semibold onb-num">v{part.active_recipe.version}</bdi>
                  <span className="onb-sub">{formatDate(part.active_recipe.released_at, language)}</span></span> : <span className="onb-muted">—</span>}</td>
                <td><div className="flex flex-wrap justify-end gap-2">
                  <Link className="onb-btn onb-btn-quiet" to={`/production/parts/${encodeURIComponent(part.part_id)}/onboard`}>
                    {part.onboarding_state === 'RELEASED' ? t.onboard : t.continueOnboarding}</Link>
                  {part.active_recipe && <Link className="onb-btn onb-btn-quiet" to={`/production/parts/${encodeURIComponent(part.part_id)}/onboard?step=release`}>{t.viewRecipe}</Link>}
                </div></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>}
    </div>
  )
}
