/**
 * Station screen copy, English and Hebrew.
 *
 * Every user-visible string of the station feature lives here so the two
 * languages stay in step. Status names (decisions, bend states, summary
 * states) come from `model/vocabulary.ts`, which mirrors the contract's
 * `DISPLAY_TOKENS`; this file adds the reason texts and the screen chrome.
 *
 * Hebrew strings have not yet been reviewed by a native speaker (same caveat
 * as the contract vocabulary).
 */
import type { Lang } from './model/types'

export type { Lang }

type Pair = { en: string; he: string }

const UNKNOWN_REASON_COPY: Record<string, Pair> = {
  OCCLUDED: { en: 'Hidden in the scan', he: 'מוסתר בסריקה' },
  EDGE_NOT_SEEN: { en: 'Edge not seen', he: 'הקצה לא נראה' },
  LOW_DENSITY: { en: 'Too few scan points', he: 'מעט מדי נקודות סריקה' },
  AMBIGUOUS_SEGMENTATION: { en: 'Flange could not be separated', he: 'לא ניתן להפריד את הכנף' },
  REGISTRATION_UNTRUSTED: { en: 'Alignment not trusted', he: 'היישור אינו אמין' },
  AMBIGUOUS_POSE: { en: 'Symmetric part, bend identity unclear', he: 'חלק סימטרי, זיהוי הכיפוף לא ודאי' },
  TIMEOUT: { en: 'Timed out', he: 'חריגת זמן' },
  UNSUPPORTED_FEATURE: { en: 'Feature not supported', he: 'מאפיין שאינו נתמך' },
  INVALID_INSPECTION: { en: 'Inspection invalid', he: 'הבדיקה לא תקפה' },
  BEND_STATE_UNKNOWN: { en: 'Bend state undetermined', he: 'מצב הכיפוף לא הוכרע' },
  ADJACENT_BEND_NOT_MEASURED: { en: 'Neighbouring bend not measured', he: 'הכיפוף הסמוך לא נמדד' },
  NOMINAL_UNAVAILABLE: { en: 'No nominal value', he: 'אין ערך נומינלי' },
  NO_TOLERANCE: { en: 'No tolerance set', he: 'לא הוגדר טולרנס' },
}

const NOT_APPLICABLE_REASON_COPY: Record<string, Pair> = {
  BEND_NOT_YET_FORMED: { en: 'Not yet formed', he: 'טרם כופף' },
  NOT_INSPECTED_BY_RECIPE: { en: 'Not inspected', he: 'לא נבדק' },
  NOT_APPLICABLE_AT_STAGE: { en: 'Not at this stage', he: 'לא בשלב זה' },
}

const STAGE_STATUS_COPY: Record<string, Pair> = {
  CONSISTENT: { en: 'Matches the bend sequence', he: 'תואם לסדר הכיפופים' },
  STAGE_MISMATCH: { en: 'Declared stage differs from the scan', he: 'השלב שהוצהר שונה מהסריקה' },
  OUT_OF_ORDER_SUSPECTED: { en: 'Bends formed out of order', he: 'כיפופים שלא לפי הסדר' },
  PARTIAL_BEND_FOUND: { en: 'A bend is only partly formed', he: 'כיפוף שכופף חלקית' },
  FINAL_INCOMPLETE: { en: 'Final check, but bends are missing', he: 'בדיקה סופית, אך חסרים כיפופים' },
  UNDETERMINED: { en: 'Stage undetermined', he: 'השלב לא הוכרע' },
  AMBIGUOUS_POSE: { en: 'Symmetric part, pose unclear', he: 'חלק סימטרי, המנח לא ודאי' },
  NOT_EVALUATED: { en: 'Stage not evaluated', he: 'השלב לא נבדק' },
}

const VALIDITY_COPY: Record<string, Pair> = {
  VALID: { en: 'Valid', he: 'תקפה' },
  INSUFFICIENT_CAPTURE: { en: 'Not enough of the part was scanned', he: 'לא נסרק מספיק מהחלק' },
  UNTRUSTED_REGISTRATION: { en: 'Scan could not be aligned to the CAD', he: 'לא ניתן ליישר את הסריקה לשרטוט' },
  UNIT_MISMATCH_SUSPECTED: { en: 'Scan units look wrong', he: 'יחידות המידה בסריקה נראות שגויות' },
  UNSUPPORTED_INPUT: { en: 'Scan file not supported', he: 'קובץ הסריקה אינו נתמך' },
  WRONG_PART_SUSPECTED: { en: 'Scan does not match this part', he: 'הסריקה אינה תואמת לחלק' },
  MIRRORED_PART_SUSPECTED: { en: 'Mirrored part suspected', he: 'חשד לחלק הפוך (מראה)' },
  TIMEOUT: { en: 'Inspection timed out', he: 'הבדיקה חרגה מהזמן' },
}

/** Why the heatmap is not drawn (viewer_assets.heatmap_unavailable_reason and local integrity codes). */
const HEATMAP_REASON_COPY: Record<string, Pair> = {
  STAGE_MODEL_UNDETERMINED: { en: 'The bending stage of the scan was not decoded', he: 'שלב הכיפוף של הסריקה לא זוהה' },
  UNAVAILABLE: { en: 'Not available for this scan', he: 'לא זמינה עבור סריקה זו' },
  HEATMAP_NOT_PUBLISHED: { en: 'Not available for this scan', he: 'לא זמינה עבור סריקה זו' },
  VIEWER_SCAN_UNAVAILABLE: { en: 'No scan points for this result', he: 'אין נקודות סריקה לתוצאה זו' },
  VIEWER_DEVIATIONS_MISSING: { en: 'Deviation data is missing', he: 'נתוני הסטייה חסרים' },
  VIEWER_DEVIATIONS_INVALID: { en: 'Deviation data is damaged', he: 'נתוני הסטייה פגומים' },
  DEVIATIONS_NOT_LOADED: { en: 'Deviation data could not be loaded', he: 'לא ניתן לטעון את נתוני הסטייה' },
  SCAN_NOT_LOADED: { en: 'Scan points could not be loaded', he: 'לא ניתן לטעון את נקודות הסריקה' },
}

const JOB_STAGE_COPY: Record<string, Pair> = {
  queued: { en: 'Scan received, waiting', he: 'הסריקה התקבלה, ממתינה' },
  preparing: { en: 'Preparing', he: 'בהכנה' },
  engine: { en: 'Measuring', he: 'מודד' },
  postprocess: { en: 'Building report', he: 'מפיק דוח' },
  done: { en: 'Done', he: 'הסתיים' },
}

const ISSUE_REASON_COPY: Record<string, Pair> = {
  UNPARSEABLE_FILENAME: { en: 'File name has no part number', he: 'בשם הקובץ אין מק"ט' },
  UNMATCHED_SKU: { en: 'Part number not released for inspection', he: 'המק"ט לא שוחרר לבדיקה' },
  AMBIGUOUS_SKU: { en: 'File name matches several parts', he: 'שם הקובץ תואם כמה חלקים' },
  REVISION_MISMATCH: { en: 'Revision differs from the released recipe', he: 'הגרסה שונה מהמתכון ששוחרר' },
  PART_NOT_READY: { en: 'Part recipe not ready', he: 'מתכון החלק אינו מוכן' },
  UNSUPPORTED_FORMAT: { en: 'Unsupported scan format', he: 'פורמט סריקה לא נתמך' },
  PLY_HEADER_INVALID: { en: 'Scan file is damaged', he: 'קובץ הסריקה פגום' },
  PLY_TRUNCATED: { en: 'Scan file is incomplete', he: 'קובץ הסריקה חלקי' },
  STL_INVALID: { en: 'Scan file is damaged', he: 'קובץ הסריקה פגום' },
  EMPTY_SCAN: { en: 'Scan file is empty', he: 'קובץ הסריקה ריק' },
  SIDECAR_INVALID: { en: 'Scan sidecar file is invalid', he: 'קובץ הנלווה לסריקה אינו תקין' },
  READ_ERROR: { en: 'Scan file could not be read', he: 'לא ניתן לקרוא את קובץ הסריקה' },
  WORKER_TIMEOUT: { en: 'Measurement took too long', he: 'המדידה ארכה יותר מדי' },
  WORKER_RSS_LIMIT: { en: 'Measurement ran out of memory', he: 'למדידה לא הספיק זיכרון' },
  ENGINE_OUTPUT_INVALID: { en: 'Measurement output was rejected', he: 'תוצאת המדידה נדחתה' },
  CANCELLED: { en: 'Inspection cancelled', he: 'הבדיקה בוטלה' },
}

export const UI = {
  title: { en: 'Press brake station', he: 'עמדת מכופפת' },
  station: { en: 'Station', he: 'עמדה' },
  serial: { en: 'Serial', he: 'סידורי' },
  recipe: { en: 'Recipe', he: 'מתכון' },
  job: { en: 'Job', he: 'עבודה' },
  connLive: { en: 'Live', he: 'מחובר' },
  connPolling: { en: 'Polling', he: 'מתעדכן' },
  connOffline: { en: 'Offline', he: 'מנותק' },
  updated: { en: 'Updated', he: 'עודכן' },
  langToggle: { en: 'עברית', he: 'English' },
  langToggleLabel: { en: 'Switch to Hebrew', he: 'החלפה לאנגלית' },
  operatorAssist: {
    en: 'Operator assist – not an automatic accept/reject',
    he: 'סיוע למפעיל – אינו קבלה או דחייה אוטומטית',
  },
  requiredChecks: { en: 'required checks', he: 'בדיקות נדרשות' },
  noBendsFormed: { en: 'No bends formed yet – nothing to check', he: 'עדיין לא בוצע כיפוף – אין מה לבדוק' },
  noRequiredChecks: { en: 'No required checks', he: 'אין בדיקות נדרשות' },
  countPass: { en: 'pass', he: 'תקין' },
  countFail: { en: 'fail', he: 'חריגה' },
  countReview: { en: 'review', he: 'לבדיקה' },
  countUnknown: { en: 'undetermined', he: 'לא הוכרע' },
  worst: { en: 'Worst', he: 'החריגה הגדולה' },
  bendsFormed: { en: 'bends formed', he: 'כיפופים בוצעו' },
  of: { en: 'of', he: 'מתוך' },
  stage: { en: 'Stage', he: 'שלב' },
  stageUnknown: { en: 'Stage ?', he: 'שלב ?' },
  finalInspection: { en: 'Final inspection', he: 'בדיקה סופית' },
  inProcessInspection: { en: 'In-process inspection', he: 'בדיקה בתהליך' },
  move: { en: 'Move', he: 'מהלך' },
  colBend: { en: 'Bend', he: 'כיפוף' },
  colFlangeA: { en: 'Flange A', he: 'כנף A' },
  colFlangeB: { en: 'Flange B', he: 'כנף B' },
  colAngle: { en: 'Angle', he: 'זווית' },
  colState: { en: 'State', he: 'מצב' },
  expected: { en: 'exp', he: 'נומ׳' },
  actual: { en: 'act', he: 'בפועל' },
  tol: { en: 'tol', he: 'טול׳' },
  bendLine: { en: 'Bend line', he: 'קו כיפוף' },
  backgauge: { en: 'Backgauge', he: 'מעצור אחורי' },
  measuredLength: { en: 'Measured', he: 'נמדד' },
  notMeasured: { en: 'Not measured', he: 'לא נמדד' },
  noValue: { en: '—', he: '—' },
  generalMeasurements: { en: 'General measurements', he: 'מידות כלליות' },
  unplannedCreases: { en: 'Unplanned crease found – review the part', he: 'נמצא קיפול לא מתוכנן – יש לבדוק את החלק' },
  layers: { en: 'Layers', he: 'שכבות' },
  layerCad: { en: 'CAD', he: 'שרטוט' },
  layerGhost: { en: 'Finished part', he: 'חלק גמור' },
  layerScan: { en: 'Scan', he: 'סריקה' },
  layerHeatmap: { en: 'Heatmap', he: 'מפת סטיות' },
  layerLabels: { en: 'Bend labels', he: 'תוויות' },
  layerOutOnly: { en: 'Problems only', he: 'רק חריגות' },
  focusWorst: { en: 'Worst first', he: 'החריגה הגדולה' },
  focusNext: { en: 'Next worst', he: 'החריגה הבאה' },
  fitAll: { en: 'Fit part', he: 'כל החלק' },
  heatmapTitle: { en: 'Deviation, mm', he: 'סטייה, מ"מ' },
  heatmapUnder: { en: 'under', he: 'חסר' },
  heatmapOver: { en: 'over', he: 'עודף' },
  heatmapNoValue: { en: 'no value', he: 'ללא ערך' },
  heatmapUnavailable: { en: 'Heatmap unavailable', he: 'מפת הסטיות אינה זמינה' },
  heatmapLoading: { en: 'Loading deviations…', he: 'טוען סטיות…' },
  cadFinishedFallback: {
    en: 'CAD: finished part (scan stage not decoded)',
    he: 'שרטוט: החלק הגמור (שלב הסריקה לא זוהה)',
  },
  noScanPoints: { en: 'No scan points for this result', he: 'אין נקודות סריקה לתוצאה זו' },
  heatmapCountMismatch: { en: 'deviation data does not match the scan points', he: 'נתוני הסטייה אינם תואמים לנקודות הסריקה' },
  loading3d: { en: 'Loading 3D view…', he: 'טוען תצוגה תלת־ממדית…' },
  loadingResult: { en: 'Loading the scan result…', he: 'טוען את תוצאת הסריקה…' },
  connecting: { en: 'Connecting to the line server…', he: 'מתחבר לשרת הקו…' },
  viewerError: { en: '3D view could not load', he: 'לא ניתן לטעון את התצוגה התלת־ממדית' },
  noWebgl: { en: 'This screen cannot draw 3D (WebGL is off)', he: 'המסך אינו מציג תלת־ממד (WebGL כבוי)' },
  downloadPdf: { en: 'Download PDF', he: 'הורדת PDF' },
  downloading: { en: 'Downloading…', he: 'מוריד…' },
  pdfNotReady: { en: 'Report not ready', he: 'הדוח טרם מוכן' },
  pdfFailed: { en: 'Download failed, try again', he: 'ההורדה נכשלה, נסו שוב' },
  history: { en: 'Recent scans', he: 'סריקות אחרונות' },
  historyEmpty: { en: 'No earlier scans yet', he: 'אין עדיין סריקות קודמות' },
  viewingEarlier: { en: 'Viewing an earlier scan', he: 'מוצגת סריקה קודמת' },
  backToLive: { en: 'Back to live', he: 'חזרה לתצוגה חיה' },
  newerAvailable: { en: 'A newer scan is ready', he: 'סריקה חדשה מוכנה' },
  staleMeasuring: { en: 'Measuring', he: 'נמדד כעת' },
  stalePrevious: { en: 'Below: previous result,', he: 'למטה: התוצאה הקודמת,' },
  staleOffline: { en: 'Connection lost since', he: 'החיבור אבד מאז' },
  staleOutdated: { en: 'The result below may be outdated', he: 'ייתכן שהתוצאה שלמטה אינה עדכנית' },
  invalidNoBends: {
    en: 'Inspection invalid – no bend was measured. Rescan the part.',
    he: 'הבדיקה לא תקפה – אף כיפוף לא נמדד. יש לסרוק שוב את החלק.',
  },
  processing: { en: 'New scan', he: 'סריקה חדשה' },
  elapsed: { en: 'elapsed', he: 'זמן' },
  idle: { en: 'Waiting for the next scan', he: 'ממתין לסריקה הבאה' },
  stalled: { en: 'No progress for', he: 'אין התקדמות כבר' },
  stalledHint: { en: 'call the engineer if it does not finish', he: 'אם הבדיקה לא מסתיימת, יש לקרוא למהנדס' },
  issueQuarantine: { en: 'Scan held back', he: 'סריקה הועברה להסגר' },
  issueFailed: { en: 'Inspection failed', he: 'הבדיקה נכשלה' },
  rescanOrCall: { en: 'Rescan / call engineer', he: 'לסרוק שוב / לקרוא למהנדס' },
  waitingFirst: { en: 'No inspection on this station yet. Results appear here as soon as a scan is measured.', he: 'עדיין אין בדיקה בעמדה זו. התוצאות יופיעו כאן מיד לאחר מדידת סריקה.' },
  resultRefusedShort: { en: 'Result refused', he: 'התוצאה נדחתה' },
  resultRejected: { en: 'The latest result failed its consistency check and is not shown. Call the engineer.', he: 'התוצאה האחרונה נכשלה בבדיקת העקביות ואינה מוצגת. יש לקרוא למהנדס.' },
  authTitle: { en: 'Connect this station', he: 'חיבור העמדה' },
  authBody: { en: 'Enter the station token issued by the line administrator.', he: 'הזינו את אסימון העמדה שהנפיק מנהל הקו.' },
  authRejected: { en: 'The saved token was not accepted. Enter a new station token.', he: 'האסימון השמור לא התקבל. הזינו אסימון עמדה חדש.' },
  authInvalidFormat: { en: 'A station token starts with sdt_', he: 'אסימון עמדה מתחיל ב-sdt_' },
  authTokenLabel: { en: 'Station token', he: 'אסימון עמדה' },
  authConnect: { en: 'Connect', he: 'חיבור' },
  authForbidden: { en: 'This token cannot read this station.', he: 'לאסימון זה אין הרשאה לעמדה זו.' },
  notConfigured: { en: 'The production line is not configured on this server.', he: 'קו הייצור אינו מוגדר בשרת זה.' },
  feedError: { en: 'Cannot reach the line server. Retrying.', he: 'אין חיבור לשרת הקו. מנסה שוב.' },
  stateFormed: { en: 'Formed', he: 'כופף' },
  // --- bender view (station v2) ---
  bendsDone: { en: 'bends done', he: 'כיפופים בוצעו' },
  partStage: { en: 'Stage', he: 'שלב' },
  colFlangeADelta: { en: 'Flange A', he: 'כנף A' },
  colFlangeBDelta: { en: 'Flange B', he: 'כנף B' },
  colAngleDelta: { en: 'Angle', he: 'זווית' },
  colVerdict: { en: 'Result', he: 'תוצאה' },
  deltaMm: { en: 'Δ mm', he: 'Δ מ"מ' },
  deltaDeg: { en: 'Δ °', he: 'Δ °' },
  measuredShort: { en: 'meas.', he: 'נמדד' },
  expectedLong: { en: 'Expected', he: 'צפוי' },
  actualLong: { en: 'Actual', he: 'בפועל' },
  deviationLong: { en: 'Off by', he: 'סטייה' },
  toleranceLong: { en: 'Tolerance', he: 'טולרנס' },
  barKeyZone: { en: 'tolerance', he: 'טולרנס' },
  barKeyMark: { en: 'measured', he: 'נמדד' },
  barKeyWhisker: { en: 'measurement uncertainty', he: 'אי־ודאות המדידה' },
  tapBendHint: { en: 'Tap a bend to see expected vs actual on the part', he: 'הקישו על כיפוף כדי לראות צפוי מול בפועל על החלק' },
  closeCallout: { en: 'Close', he: 'סגירה' },
  bendsToFix: { en: 'to fix', he: 'לתיקון' },
  bendsToCheck: { en: 'to check', he: 'לבדיקה' },
  bendsOk: { en: 'OK', he: 'תקינים' },
  bendsUnsure: { en: 'not measured', he: 'לא נמדדו' },
  bendsNotBent: { en: 'not bent yet', he: 'טרם כופפו' },
  bendsWord: { en: 'bends', he: 'כיפופים' },
  alsoPosition: { en: 'Bend line position', he: 'מיקום קו הכיפוף' },
  logoAlt: { en: 'Sherman', he: 'שרמן' },
  ofMoves: { en: 'Move', he: 'מהלך' },
  // --- zero-shot overlay (sherman_zeroshot_overlay_v1): detections, never measurements ---
  zsTitle: { en: 'Detections (unverified)', he: 'זיהויים (לא מאומתים)' },
  zsNote: {
    en: 'Found in the scan by the AI detection, not measured against the CAD. Never a pass.',
    he: 'זוהו בסריקה בזיהוי האוטומטי, ללא מדידה מול השרטוט. לעולם אינם אישור.',
  },
  zsRefusedNote: {
    en: 'Detected bends, unverified: the scan did not match the part.',
    he: 'כיפופים שזוהו, לא מאומתים: הסריקה לא הותאמה לחלק.',
  },
  zsKindWithheld: { en: 'Position not measured: detected line', he: 'מיקום לא נמדד: קו שזוהה' },
  zsKindUnexpected: { en: 'Unexpected bend', he: 'כיפוף לא צפוי' },
  zsKindRefused: { en: 'Detected bend', he: 'כיפוף שזוהה' },
  zsUnverified: { en: 'unverified', he: 'לא מאומת' },
  zsCheck: { en: 'check', he: 'לבדוק' },
  zsReasonUnexpected: { en: 'Unexpected bend found: check the part', he: 'נמצא כיפוף לא צפוי: יש לבדוק את החלק' },
  zsLength: { en: 'mm long', he: 'מ"מ אורך' },
  zsAngleJoggle: { en: 'joggle, angle not measured', he: "קיזוז, הזווית לא נמדדה" },
  zsReasonAngle: { en: 'Independent angle disagrees with the FAIL: check the angle', he: 'הזווית העצמאית אינה תואמת לחריגה: יש לבדוק את הזווית' },
  // --- left/right-sensitive part (display only, until the engine checks it) ---
  chiralityCheck: { en: 'Check left/right variant by eye', he: 'בדקו בעין את גרסת ימין/שמאל' },
  chiralityCheckDetail: {
    en: 'Left/right-sensitive part: the line does not yet tell it from its mirror image.',
    he: 'חלק רגיש לימין/שמאל: הקו עדיין לא מבחין בינו לבין תמונת הראי שלו.',
  },
  // --- part picker (held-back scans) ---
  choosePart: { en: 'Choose part', he: 'בחירת חלק' },
  heldDetails: { en: 'Details', he: 'פרטים' },
  heldCount: { en: 'held back', he: 'בהסגר' },
  pickerTitle: { en: 'Held-back scan: choose the part', he: 'סריקה בהסגר: בחירת החלק' },
  pickerHeldList: { en: 'Held-back scans', he: 'סריקות בהסגר' },
  pickerWhy: { en: 'Why it was held', he: 'סיבת ההסגר' },
  pickerFile: { en: 'File', he: 'קובץ' },
  pickerReceived: { en: 'Received', he: 'התקבלה' },
  pickerParsed: { en: 'Read from the name', he: 'נקרא מהשם' },
  pickerScanStats: { en: 'Scan', he: 'סריקה' },
  pickerPoints: { en: 'points', he: 'נקודות' },
  pickerSuggested: { en: 'Suggested', he: 'מוצע' },
  pickerSearchLabel: { en: 'Search released parts (part number or name)', he: 'חיפוש בחלקים ששוחררו (מק"ט או שם)' },
  pickerNoParts: { en: 'No released part matches', he: 'אין חלק משוחרר תואם' },
  pickerNotReady: { en: 'Not ready', he: 'לא מוכן' },
  pickerRevision: { en: 'Rev.', he: 'גרסה' },
  pickerVersion: { en: 'Recipe version', he: 'גרסת מתכון' },
  pickerVersionActive: { en: 'active', he: 'פעילה' },
  pickerVersionOld: { en: 'not active', he: 'לא פעילה' },
  pickerStage: { en: 'Scan stage', he: 'שלב הסריקה' },
  stageFinal: { en: 'Final (all bends done)', he: 'סופית (כל הכיפופים בוצעו)' },
  stageInProcess: { en: 'In process (part-way)', he: 'בתהליך (חלקית)' },
  pickerBendsFormed: { en: 'Bends done so far', he: 'כיפופים שבוצעו עד כה' },
  pickerBendsInfer: { en: 'let the inspection detect', he: 'הבדיקה תזהה' },
  pickerStageDefault: { en: 'recipe default', he: 'ברירת המחדל של המתכון' },
  pickerNext: { en: 'Next', he: 'המשך' },
  pickerBack: { en: 'Back', he: 'חזרה' },
  pickerCancel: { en: 'Close', he: 'סגירה' },
  pickerConfirmTitle: { en: 'Inspect this scan as', he: 'הסריקה תיבדק כחלק' },
  pickerConfirm: { en: 'Confirm and run', he: 'אישור והרצה' },
  pickerConfirmNote: {
    en: 'The inspection still checks that the scan is this part. A wrong part is never passed.',
    he: 'הבדיקה עדיין מוודאת שהסריקה היא של חלק זה. חלק שגוי לעולם לא יאושר.',
  },
  pickerAssigning: { en: 'Starting…', he: 'מתחיל…' },
  pickerAssigned: { en: 'Part assigned. The inspection is starting.', he: 'החלק הוקצה. הבדיקה מתחילה.' },
  pickerDismiss: { en: 'Dismiss scan', he: 'ביטול הסריקה' },
  pickerDismissTitle: { en: 'Dismiss this scan', he: 'ביטול סריקה זו' },
  pickerDismissReason: { en: 'Reason', he: 'סיבה' },
  pickerDismissNote: { en: 'Note', he: 'הערה' },
  pickerDismissNoteHint: { en: 'required for "Other"', he: 'חובה עבור "אחר"' },
  pickerDismissDo: { en: 'Dismiss', he: 'ביטול' },
  pickerDismissed: { en: 'Scan dismissed.', he: 'הסריקה בוטלה.' },
  pickerNotResolvable: {
    en: 'The file itself is unusable. It cannot be assigned: export the scan again, then dismiss this one.',
    he: 'הקובץ עצמו אינו שמיש. לא ניתן להקצות אותו: יש לייצא שוב את הסריקה ולבטל את זו.',
  },
  pickerNeedOperator: { en: 'An operator must sign in to assign or dismiss.', he: 'נדרשת כניסת מפעיל כדי להקצות או לבטל.' },
  pickerUser: { en: 'User name', he: 'שם משתמש' },
  pickerPassword: { en: 'Password', he: 'סיסמה' },
  pickerSignIn: { en: 'Sign in', he: 'כניסה' },
  pickerSignedInAs: { en: 'Operator', he: 'מפעיל' },
  pickerSignInFailed: { en: 'Sign-in failed', he: 'הכניסה נכשלה' },
  pickerNotOperator: { en: 'This user cannot operate the line', he: 'למשתמש זה אין הרשאת הפעלה' },
  pickerError: { en: 'Not done', he: 'לא בוצע' },
  pickerEmpty: { en: 'No scan is held back.', he: 'אין סריקות בהסגר.' },
  assignedByOperator: { en: 'Part assigned by operator', he: 'החלק הוקצה ע"י מפעיל' },
} satisfies Record<string, Pair>

export type UiKey = keyof typeof UI

export function t(key: UiKey, lang: Lang): string {
  return UI[key][lang]
}

function lookup(table: Record<string, Pair>, code: string, lang: Lang): string {
  const pair = table[code]
  return pair ? pair[lang] : code
}

export const unknownReasonText = (code: string, lang: Lang) => lookup(UNKNOWN_REASON_COPY, code, lang)
export const notApplicableReasonText = (code: string, lang: Lang) => lookup(NOT_APPLICABLE_REASON_COPY, code, lang)
export const stageStatusText = (code: string, lang: Lang) => lookup(STAGE_STATUS_COPY, code, lang)
export const validityText = (code: string, lang: Lang) => lookup(VALIDITY_COPY, code, lang)
export const jobStageText = (code: string, lang: Lang) => lookup(JOB_STAGE_COPY, code, lang)
export const issueReasonText = (code: string, lang: Lang) => lookup(ISSUE_REASON_COPY, code, lang)

/**
 * Operator wording for a heatmap-unavailable reason: heatmap codes, validity
 * states (`VALIDITY_<STATE>`), then characteristic reasons; an unknown code
 * gets a generic sentence, never the raw enum text.
 */
export function heatmapReasonText(code: string, lang: Lang): string {
  if (code in HEATMAP_REASON_COPY) return HEATMAP_REASON_COPY[code][lang]
  if (code.startsWith('VALIDITY_') && code.slice(9) in VALIDITY_COPY) return validityText(code.slice(9), lang)
  if (code in VALIDITY_COPY) return validityText(code, lang)
  if (code in UNKNOWN_REASON_COPY || code in NOT_APPLICABLE_REASON_COPY) return reasonText(code, lang)
  return HEATMAP_REASON_COPY.UNAVAILABLE[lang]
}

/** Reason text for any characteristic reason code (UNKNOWN or NOT_APPLICABLE). */
export function reasonText(code: string, lang: Lang): string {
  if (code in UNKNOWN_REASON_COPY) return unknownReasonText(code, lang)
  if (code in NOT_APPLICABLE_REASON_COPY) return notApplicableReasonText(code, lang)
  return code
}

/** Verdict words (plan mapping PASS = OK, REVIEW = CHECK, FAIL = FIX). */
const VERDICT_COPY: Record<string, Pair> = {
  OK: { en: 'OK', he: 'תקין' },
  CHECK: { en: 'CHECK', he: 'לבדוק' },
  FIX: { en: 'FIX', he: 'לתקן' },
  UNSURE: { en: 'Not measured', he: 'לא נמדד' },
  NOT_BENT: { en: 'Not bent yet', he: 'טרם כופף' },
  NONE: { en: '—', he: '—' },
  OK_SO_FAR: { en: 'OK so far', he: 'תקין עד כה' },
  RESCAN: { en: 'Rescan', he: 'לסרוק שוב' },
}

/** What the bender does next, per verdict. */
const ACTION_COPY: Record<string, Pair> = {
  OK: { en: 'In tolerance. Carry on.', he: 'בטולרנס. ממשיכים.' },
  CHECK: {
    en: 'Near the limit, or the scan is not certain. Check with a gauge before the next bend.',
    he: 'קרוב לגבול, או שהסריקה אינה ודאית. לבדוק במד לפני הכיפוף הבא.',
  },
  FIX: {
    en: 'Out of tolerance. Correct it, or set the part aside and tell the shift lead.',
    he: 'מחוץ לטולרנס. לתקן, או להפריש את החלק ולעדכן את ראש המשמרת.',
  },
  UNSURE: { en: 'The scan could not measure this. Check by hand.', he: 'הסריקה לא הצליחה למדוד. לבדוק ידנית.' },
  NOT_BENT: { en: 'Nothing to check until it is bent.', he: 'אין מה לבדוק עד שיכופף.' },
  NONE: { en: 'Nothing to check on this bend.', he: 'אין מה לבדוק בכיפוף זה.' },
  OK_SO_FAR: { en: 'Every bend made so far is in tolerance.', he: 'כל הכיפופים שבוצעו עד כה בטולרנס.' },
  RESCAN: { en: 'This scan cannot be used. Rescan the part.', he: 'לא ניתן להשתמש בסריקה זו. יש לסרוק שוב את החלק.' },
}

/** Direction of an out-of-tolerance value (FAIL only). */
const DIRECTION_COPY: Record<string, Pair> = {
  OVERBENT: { en: 'bent too far', he: 'כיפוף יתר' },
  UNDERBENT: { en: 'not bent enough', he: 'כיפוף חסר' },
  TOO_LONG: { en: 'too long', he: 'ארוך מדי' },
  TOO_SHORT: { en: 'too short', he: 'קצר מדי' },
}

export const verdictText = (code: string, lang: Lang) => lookup(VERDICT_COPY, code, lang)
export const actionText = (code: string, lang: Lang) => lookup(ACTION_COPY, code, lang)
export const directionText = (code: string, lang: Lang) => lookup(DIRECTION_COPY, code, lang)

/** Why a pick was refused (assign / dismiss error codes from the line API). */
const PICKER_ERROR_COPY: Record<string, Pair> = {
  ALREADY_RESOLVED: { en: 'Another operator already handled this scan', he: 'מפעיל אחר כבר טיפל בסריקה זו' },
  SCAN_NOT_QUARANTINED: { en: 'This scan is no longer held back', he: 'סריקה זו כבר אינה בהסגר' },
  RECIPE_NOT_ACTIVE: { en: 'A newer recipe version was released; choose again', he: 'שוחררה גרסת מתכון חדשה; יש לבחור שוב' },
  RECIPE_RETIRED: { en: 'This recipe was retired', he: 'המתכון הוצא משימוש' },
  PART_NOT_READY: { en: 'This part is not ready for inspection', he: 'החלק אינו מוכן לבדיקה' },
  NOT_RESOLVABLE: { en: 'The file itself is unusable', he: 'הקובץ עצמו אינו שמיש' },
  STAGE_INDEX_REQUIRED: { en: 'Enter how many bends are done', he: 'יש להזין כמה כיפופים בוצעו' },
  STAGE_INDEX_INVALID: { en: 'That bend count does not fit this part', he: 'מספר הכיפופים אינו מתאים לחלק' },
  CAPABILITY_REQUIRED: { en: 'An operator must sign in', he: 'נדרשת כניסת מפעיל' },
}

/** Where a suggested part came from. */
const SUGGESTION_SOURCE_COPY: Record<string, Pair> = {
  RESOLVER_CANDIDATE: { en: 'named in the file', he: 'מופיע בשם הקובץ' },
  FILENAME_SKU: { en: 'part number in the file', he: 'המק"ט שבשם הקובץ' },
  FILENAME_FUZZY: { en: 'similar to the file name', he: 'דומה לשם הקובץ' },
}

/** Operator dismiss reasons. */
const DISMISS_REASON_COPY: Record<string, Pair> = {
  RESCAN: { en: 'Will rescan the part', he: 'החלק ייסרק שוב' },
  NOT_A_PART: { en: 'Not a production part', he: 'לא חלק ייצור' },
  DUPLICATE: { en: 'Duplicate scan', he: 'סריקה כפולה' },
  WRONG_STATION: { en: 'Wrong station', he: 'עמדה שגויה' },
  OTHER: { en: 'Other', he: 'אחר' },
}

export const pickerErrorText = (code: string | null, lang: Lang) => (code && code in PICKER_ERROR_COPY ? PICKER_ERROR_COPY[code][lang] : UI.pickerError[lang])
export const suggestionSourceText = (code: string, lang: Lang) => lookup(SUGGESTION_SOURCE_COPY, code, lang)
export const dismissReasonText = (code: string, lang: Lang) => lookup(DISMISS_REASON_COPY, code, lang)
