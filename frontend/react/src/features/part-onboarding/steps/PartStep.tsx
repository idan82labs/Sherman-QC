/** Step 1: part details (read from the catalog; the SKU is shown in canonical form). */
import { canonicalSkuPreview, mergedPart } from '../model/checklist'
import { ReasonRow } from '../components/common'
import type { StepProps } from './stepProps'

export default function PartStep({ snapshot, t, language }: StepProps) {
  const part = mergedPart(snapshot)
  const thickness = snapshot.releaseDraft?.request?.thickness_mm ?? part?.thickness_mm ?? null
  if (!part) return <p className="onb-muted">{t.loading}</p>
  const sku = canonicalSkuPreview(part.part_number)
  const rows: Array<[string, string | null, boolean]> = [
    [t.sku, snapshot.onboarding?.sku ?? (sku.ok ? sku.sku : null), true],
    [t.partNumber, part.part_number, true],
    [t.partName, part.part_name, false],
    [t.revision, part.revision, true],
    [t.material, part.material, false],
    [t.recipeThickness, thickness != null ? String(thickness) : null, true],
    [t.customer, part.customer ?? null, false],
  ]
  return (
    <div className="onb-stack">
      <dl className="onb-spec" data-part-details>
        {rows.map(([label, value, ltr]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd className={value ? undefined : 'onb-muted'}>{value ? (ltr ? <bdi dir="ltr" className="onb-num">{value}</bdi> : value) : t.noValue}</dd>
          </div>
        ))}
      </dl>
      {!sku.ok && <ReasonRow code="SKU_INVALID" language={language} />}
      <p className="onb-help">{t.partDetailsHint}</p>
    </div>
  )
}
