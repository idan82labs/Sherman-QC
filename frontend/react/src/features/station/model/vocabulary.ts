/**
 * Display tokens mirrored from `contracts/production/vocabulary.py`
 * (`DISPLAY_TOKENS`). `scripts/check-station-view-model.mjs` compares this
 * table against the Python one, so edits must be made in both places.
 *
 * Colour rule (binding): PASS green, FAIL red, REVIEW amber, UNKNOWN neutral
 * grey, NOT_YET_FORMED / NOT_APPLICABLE muted slate. STAGE_IN_TOLERANCE is
 * blue "info"; only ALL_MEASURED_IN_TOLERANCE is green. An UNKNOWN is never
 * shown with a pass or fail colour.
 */

export type Tone = 'pass' | 'fail' | 'review' | 'unknown' | 'inactive' | 'neutral' | 'info'

export interface DisplayToken {
  tone: Tone
  color: string
  en: string
  he: string
}

export const DISPLAY_TOKENS: Readonly<Record<string, DisplayToken>> = Object.freeze({
  // decisions
  PASS: { tone: 'pass', color: '#16A34A', en: 'Pass', he: 'תקין' },
  FAIL: { tone: 'fail', color: '#DC2626', en: 'Fail', he: 'חריגה' },
  REVIEW: { tone: 'review', color: '#F59E0B', en: 'Review', he: 'לבדיקה' },
  UNKNOWN: { tone: 'unknown', color: '#6B7280', en: 'Undetermined', he: 'לא הוכרע' },
  NOT_APPLICABLE: { tone: 'inactive', color: '#94A3B8', en: 'Not applicable', he: 'לא רלוונטי' },
  // bend states
  FORMED: { tone: 'neutral', color: '#334155', en: 'Formed', he: 'כופף' },
  NOT_YET_FORMED: { tone: 'inactive', color: '#94A3B8', en: 'Not yet formed', he: 'טרם כופף' },
  PARTIALLY_FORMED: { tone: 'review', color: '#F59E0B', en: 'Partially formed', he: 'כופף חלקית' },
  UNEXPECTED_FORMED: { tone: 'review', color: '#F59E0B', en: 'Formed out of order', he: 'כופף שלא לפי הסדר' },
  // summary states
  ALL_MEASURED_IN_TOLERANCE: { tone: 'pass', color: '#16A34A', en: 'All measured in tolerance', he: 'כל המידות בטולרנס' },
  STAGE_IN_TOLERANCE: {
    tone: 'info',
    color: '#2563EB',
    en: 'Stage in tolerance (part not complete)',
    he: 'השלב בטולרנס (החלק טרם הושלם)',
  },
  OUT_OF_TOLERANCE_FOUND: { tone: 'fail', color: '#DC2626', en: 'Out of tolerance found', he: 'נמצאה חריגה' },
  REVIEW_REQUIRED: { tone: 'review', color: '#F59E0B', en: 'Review required', he: 'נדרשת בדיקה' },
  INCOMPLETE: { tone: 'unknown', color: '#6B7280', en: 'Incomplete', he: 'לא הושלם' },
  INVALID_INSPECTION: { tone: 'unknown', color: '#6B7280', en: 'Invalid inspection', he: 'בדיקה לא תקפה' },
})

export function displayToken(term: string): DisplayToken {
  const token = DISPLAY_TOKENS[term]
  if (!token) throw new Error(`no display token for ${term}`)
  return token
}
