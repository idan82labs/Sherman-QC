/** Props every wizard step receives from OnboardingWizardPage. */
import type { OnbCopy, OnbLanguage } from '../copy'
import type { StepState, VendorSummary, WizardSnapshot, WizardStepKey } from '../model/checklist'

export interface StepProps {
  partId: string
  language: OnbLanguage
  t: OnbCopy
  snapshot: WizardSnapshot
  summary: VendorSummary
  steps: StepState[]
  /** Re-read every server source the wizard depends on. */
  refresh: () => Promise<void>
  goTo: (key: WizardStepKey) => void
  /** Open the tagging studio (Vendor Setup inside the wizard, step 5). */
  openStudio: () => void
}
