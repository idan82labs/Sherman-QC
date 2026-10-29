/**
 * Production onboarding (C09). C11 mounts the three pages inside Layout:
 *
 *   /production/parts                     -> ProductionPartsPage
 *   /production/parts/new                 -> NewProductionPartPage
 *   /production/parts/:partId/onboard     -> OnboardingWizardPage (?step=part|cad|drawing|flat|bends|tolerances|review|release)
 *
 * Vendor Setup links back with `/parts/vendor-setup/:partId?return=<wizard path>`
 * (see `model/flat.ts` vendorSetupHref / safeWizardReturn).
 */
export { default as ProductionPartsPage } from './pages/ProductionPartsPage'
export { default as NewProductionPartPage } from './pages/NewProductionPartPage'
export { default as OnboardingWizardPage } from './pages/OnboardingWizardPage'
export { vendorSetupHref, wizardReturnHref } from './model/flat'
