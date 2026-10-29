import type { Point3, VendorBendMark, VendorSetupDraft, VendorSetupResponse } from '../../../types'

export interface VendorSetupPdfView {
  dataUrl: string
  width: number
  height: number
  labels: Array<{ x: number; y: number; text: string; color: string; featureId?: string }>
}

export type VendorSetupPdfViewMode = 'folded' | 'unfolded_visual'

export interface VendorSetupPdfInput {
  data: VendorSetupResponse
  draft: VendorSetupDraft
  language: 'he' | 'en'
  view: VendorSetupPdfView | null
  viewMode?: VendorSetupPdfViewMode
  additionalView?: { mode: VendorSetupPdfViewMode; view: VendorSetupPdfView }
}

type Cell = { text: string; ltr?: boolean; color?: string }
type Sheet = { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D }
type PageImage = { bytes: Uint8Array; width: number; height: number }

const WIDTH = 1684
const HEIGHT = 1191
const PIXEL_WIDTH = 3508
const PIXEL_HEIGHT = 2480
const MARGIN = 36
const CONTENT_TOP = 202
const CONTENT_BOTTOM = 1006
const INK = '#15212d'
const MUTED = '#46515d'
const BORDER = '#8894a0'
const FONT = 'Heebo, Assistant, Arial, sans-serif'
const MAX_ROWS = 200
const COLORS = ['#22d3ee', '#fb7185', '#a3e635', '#fbbf24', '#c084fc', '#60a5fa', '#f472b6', '#34d399', '#fb923c', '#e879f9', '#facc15', '#2dd4bf']

const COPY = {
  en: {
    title: 'VENDOR BEND SETUP', view: 'OPERATOR CURRENT VIEW', notToScale: 'NOT TO SCALE',
    foldedView: 'FOLDED PART - OPERATOR VIEW', unfoldedView: 'UNFOLDED PART - VISUAL GUIDE',
    unfoldedDisclaimer: 'Geometric development of the selected CAD skin only. No K-factor or bend allowance is applied. NOT a cutting blank or manufacturing pattern. CAD units are assumed to be mm and are unverified. This view is not to scale.',
    foldedReferences: 'FOLDED-PART REFERENCES - NOT FLAT-PATTERN DIMENSIONS',
    annotationKey: 'CAPTURED VIEW ANNOTATION KEY', annotation: 'Annotation', annotationDefinition: 'Supplied label',
    annotationHelp: 'Numeric annotation pointers identify captured labels only, not bend order. Direction text is included only when supplied in the captured label.',
    noView: 'No view was captured. Bend definitions and reference values are listed on the following sheets.',
    legend: 'MACHINE MOVE / BEND LEGEND', legendContinuation: 'MACHINE MOVE / BEND LEGEND - CONTINUED',
    bendTable: 'FOLDED-PART BEND SCHEDULE', measurements: 'FOLDED-PART GENERAL REFERENCE MEASUREMENTS', notes: 'OPERATOR NOTES',
    part: 'Part', revision: 'Revision', material: 'Material', issued: 'Exported', sheet: 'Sheet',
    status: 'Sequence', confirmed: 'Operator confirmed', pending: 'Not operator confirmed',
    move: 'Move', ungrouped: 'No move assigned', order: 'Order', bend: 'Bend label',
    angle: 'CAD angle / operator target (deg)', width: 'Line span (CAD units*)', depth: 'Flange depth (CAD units*)',
    sourceReview: 'Source / review', cad: 'CAD candidate', manual: 'Manual annotation', reviewed: 'Confirmed', unreviewed: 'Pending',
    generalLabel: 'Reference', definition: 'Definition / endpoints (CAD coordinates)', distance: 'Anchor distance (CAD units*)',
    target: 'Operator target (mm)', note: 'Note', points: 'Picked points', pair: 'Picked anchors associated with bends',
    noMeasurements: 'No general reference measurements were authored.', noNotes: 'No operator notes were authored.',
    noBends: 'No active bends were authored.', noGroups: 'No machine moves were authored. Unassigned bends are not implied to be separate moves.',
    reference: 'REFERENCE ONLY - NOT A RELEASED MANUFACTURING DRAWING',
    units: '* CAD units are assumed to be mm; units have not been verified. Values are nominal CAD references or operator annotations, not scan measurements.',
    safety: 'No tolerances or conformance are inferred. This sheet is not a scan/QC result, inspection acceptance, or manufacturing release. Verify units, definitions and process before use.',
    definitions: 'All dimensions refer to the folded part, not an unfolded blank. Line span = Euclidean distance between the defined bend-line endpoints. Flange depth = folded CAD tangent-to-outer-edge reference, when available. A blank value means unavailable.',
    generalDefinitions: 'All endpoints and distances remain in the folded-part CAD coordinate system; they are not flat-pattern dimensions. Anchor distance = Euclidean distance between the explicitly picked endpoints. A bend-pair reference associates the picked surface anchors with the listed bends/flanges; it is not a midpoint or nearest-edge distance. Targets are operator-entered values.',
    draftDisclosure: 'Current editable operator draft; this export may contain unsaved changes.',
    active: 'Active bends', moves: 'Authored moves', hash: 'CAD hash', draftRevision: 'Draft revision',
    sectionContinued: 'CONTINUED', noValue: '-', removedNotes: 'Removed annotation', partNotes: 'Part notes',
  },
  he: {
    title: 'דף הכנה לכיפופים לספק', view: 'המבט הנוכחי של המפעיל', notToScale: 'לא בקנה מידה',
    foldedView: 'החלק המקופל', unfoldedView: 'פריסת החלק - להמחשה בלבד',
    unfoldedDisclaimer: 'פריסה של משטח אחד מתוך ה-CAD, ללא חישוב פיצוי כיפוף. לא להשתמש בה לחיתוך או לייצור. מניחים שיחידות ה-CAD הן מ״מ; היחידות לא אומתו. המבט אינו בקנה מידה.',
    foldedReferences: 'מידות ייחוס לחלק המכופף - לא מידות לפריסת חיתוך',
    annotationKey: 'מקרא סימונים שנלכדו במבט', annotation: 'סימון', annotationDefinition: 'תווית שסופקה',
    annotationHelp: 'מצביעים מספריים מזהים את הסימונים שנלכדו בלבד, ולא את סדר הכיפופים. כיוון מוצג רק כאשר סופק בתווית שנלכדה.',
    noView: 'לא נלכד מבט. הגדרות הכיפופים וערכי הייחוס מופיעים בדפים הבאים.',
    legend: 'מקרא מהלכי מכונה וכיפופים', legendContinuation: 'מקרא מהלכי מכונה וכיפופים - המשך',
    bendTable: 'טבלת כיפופים לחלק המכופף', measurements: 'מידות ייחוס כלליות לחלק המכופף', notes: 'הערות מפעיל',
    part: 'מק״ט', revision: 'מהדורה', material: 'חומר', issued: 'מועד ייצוא', sheet: 'דף',
    status: 'סדר כיפופים', confirmed: 'אושר על ידי המפעיל', pending: 'לא אושר על ידי המפעיל',
    move: 'מהלך', ungrouped: 'ללא שיוך למהלך', order: 'סדר', bend: 'תווית כיפוף',
    angle: 'זווית CAD / יעד מפעיל (מעלות)', width: 'אורך קו (יחידות CAD*)', depth: 'עומק שפה (יחידות CAD*)',
    sourceReview: 'מקור / סקירה', cad: 'מועמד CAD', manual: 'סימון ידני', reviewed: 'אושר', unreviewed: 'ממתין',
    generalLabel: 'מידת ייחוס', definition: 'הגדרה / נקודות קצה (קואורדינטות CAD)', distance: 'מרחק עוגנים (יחידות CAD*)',
    target: 'יעד מפעיל (מ״מ)', note: 'הערה', points: 'נקודות שנבחרו', pair: 'עוגנים שנבחרו ומשויכים לכיפופים',
    noMeasurements: 'לא הוגדרו מידות ייחוס כלליות.', noNotes: 'לא נכתבו הערות מפעיל.',
    noBends: 'לא הוגדרו כיפופים פעילים.', noGroups: 'לא הוגדרו מהלכי מכונה. כיפופים ללא שיוך אינם מוצגים כמהלכים נפרדים.',
    reference: 'לייחוס בלבד - אינו שרטוט ייצור מאושר',
    units: '* מניחים שיחידות ה-CAD הן מ״מ; היחידות לא אומתו. הערכים הם ייחוס CAD נומינלי או סימוני מפעיל, ולא מדידות סריקה.',
    safety: 'לא מוסקות סבולות או התאמה. הדף אינו תוצאת סריקה/QC, אישור בדיקה או שחרור לייצור. יש לאמת יחידות, הגדרות ותהליך לפני שימוש.',
    definitions: 'כל המידות מתייחסות לחלק המכופף, לא לפריסת חיתוך. אורך קו = מרחק אוקלידי בין נקודות הקצה המוגדרות של קו הכיפוף. עומק שפה = ייחוס CAD של החלק המכופף מהמשיק לשפה החיצונית, כאשר זמין. ערך חסר אינו זמין.',
    generalDefinitions: 'כל נקודות הקצה והמרחקים נשארים במערכת הקואורדינטות של החלק המכופף; אלו אינן מידות לפריסת חיתוך. מרחק עוגנים = מרחק אוקלידי בין נקודות הקצה שנבחרו במפורש. מידת זוג כיפופים משייכת את עוגני המשטח לכיפופים/לשפות המפורטים; זה אינו מרחק בין מרכזי קווים או בין הקצוות הקרובים. היעדים מוזנים על ידי המפעיל.',
    draftDisclosure: 'טיוטת המפעיל הנוכחית הניתנת לעריכה; הייצוא עשוי לכלול שינויים שלא נשמרו.',
    active: 'כיפופים פעילים', moves: 'מהלכים שהוגדרו', hash: 'מזהה CAD', draftRevision: 'מהדורת טיוטה',
    sectionContinued: 'המשך', noValue: '-', removedNotes: 'סימון שהוסר', partNotes: 'הערות חלק',
  },
} as const

function cleanText(value: unknown): string {
  return Array.from(String(value ?? '')).filter(character => {
    const code = character.charCodeAt(0)
    return code >= 32 && code !== 127 || code === 9 || code === 10 || code === 13
  }).join('')
}

function number(value: number | null | undefined): string {
  return value != null && Number.isFinite(value) ? value.toFixed(2).replace(/\.00$/, '') : '-'
}

function point(value: Point3 | null | undefined): string {
  return value?.length === 3 && value.every(Number.isFinite) ? `(${value.map(number).join(', ')})` : '-'
}

function distance(start: Point3 | null, end: Point3 | null): number | null {
  return start?.length === 3 && end?.length === 3 && start.every(Number.isFinite) && end.every(Number.isFinite)
    ? Math.hypot(...start.map((value, index) => value - end[index])) : null
}

function filenamePart(value: string): string {
  return cleanText(value).normalize('NFKC').replace(/[\\/:*?"<>|]/g, '-').replace(/\s+/g, '-').replace(/^\.+|\.+$/g, '').slice(0, 90) || 'part'
}

function setFont(ctx: CanvasRenderingContext2D, size = 21, bold = false): void {
  ctx.font = `${bold ? 700 : 400} ${size}px ${FONT}`
}

function text(ctx: CanvasRenderingContext2D, value: string, x: number, y: number, width: number, options: { rtl?: boolean; size?: number; bold?: boolean; color?: string } = {}): void {
  setFont(ctx, options.size ?? 21, options.bold)
  ctx.fillStyle = options.color ?? INK
  ctx.direction = options.rtl ? 'rtl' : 'ltr'
  ctx.textAlign = options.rtl ? 'right' : 'left'
  ctx.textBaseline = 'top'
  ctx.fillText(cleanText(value), options.rtl ? x + width : x, y, width)
}

function wrap(ctx: CanvasRenderingContext2D, value: string, width: number, size: number, bold = false): string[] {
  setFont(ctx, size, bold)
  const lines: string[] = []
  for (const paragraph of cleanText(value).split('\n')) {
    if (!paragraph) { lines.push(''); continue }
    let line = ''
    for (const word of paragraph.split(/\s+/)) {
      if (ctx.measureText(word).width > width) {
        if (line) lines.push(line)
        let fragment = ''
        for (const character of Array.from(word)) {
          if (fragment && ctx.measureText(fragment + character).width > width) { lines.push(fragment); fragment = '' }
          fragment += character
        }
        line = fragment
      } else if (line && ctx.measureText(`${line} ${word}`).width > width) {
        lines.push(line)
        line = word
      } else {
        line = line ? `${line} ${word}` : word
      }
    }
    if (line) lines.push(line)
  }
  return lines.length ? lines : ['']
}

function paragraph(ctx: CanvasRenderingContext2D, value: string, x: number, y: number, width: number, options: { rtl?: boolean; size?: number; bold?: boolean; color?: string; lineHeight?: number } = {}): number {
  const size = options.size ?? 21
  const lineHeight = options.lineHeight ?? size * 1.45
  const lines = wrap(ctx, value, width, size, options.bold)
  lines.forEach((line, index) => text(ctx, line, x, y + index * lineHeight, width, options))
  return lines.length * lineHeight
}

function box(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, fill?: string): void {
  if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, width, height) }
  ctx.strokeStyle = BORDER
  ctx.lineWidth = 1
  ctx.strokeRect(x, y, width, height)
}

function decodeView(view: VendorSetupPdfView | null): Promise<HTMLImageElement | null> {
  if (!view) return Promise.resolve(null)
  if (!/^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=\s]+$/.test(view.dataUrl)
    || !Number.isFinite(view.width) || !Number.isFinite(view.height) || view.width <= 0 || view.height <= 0) {
    return Promise.reject(new Error('The captured view is not a valid raster image.'))
  }
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('The captured view could not be decoded.'))
    image.src = view.dataUrl
  })
}

async function jpeg(sheet: Sheet): Promise<PageImage> {
  const blob = await new Promise<Blob>((resolve, reject) => sheet.canvas.toBlob(
    result => result ? resolve(result) : reject(new Error('PDF sheet could not be rendered.')), 'image/jpeg', 0.96,
  ))
  return { bytes: new Uint8Array(await blob.arrayBuffer()), width: sheet.canvas.width, height: sheet.canvas.height }
}

/** Embeds full-resolution JPEG sheets without a PDF/font dependency or HTML injection. */
function pdf(images: PageImage[]): Blob {
  const encoder = new TextEncoder()
  const parts: BlobPart[] = []
  const offsets = [0]
  let length = 0
  const appendBytes = (bytes: Uint8Array) => { parts.push(new Uint8Array(bytes).buffer); length += bytes.byteLength }
  const append = (value: string) => appendBytes(encoder.encode(value))
  const object = (id: number, body: string) => { offsets[id] = length; append(`${id} 0 obj\n${body}\nendobj\n`) }
  append('%PDF-1.4\n')
  appendBytes(new Uint8Array([37, 226, 227, 207, 211, 10]))
  object(1, '<< /Type /Catalog /Pages 2 0 R >>')
  object(2, `<< /Type /Pages /Count ${images.length} /Kids [${images.map((_, index) => `${3 + index * 3} 0 R`).join(' ')}] >>`)
  images.forEach((image, index) => {
    const id = 3 + index * 3
    object(id, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 841.89 595.28] /Resources << /XObject << /Im0 ${id + 2} 0 R >> >> /Contents ${id + 1} 0 R >>`)
    const commands = 'q\n841.89 0 0 595.28 0 0 cm\n/Im0 Do\nQ\n'
    object(id + 1, `<< /Length ${encoder.encode(commands).byteLength} >>\nstream\n${commands}endstream`)
    offsets[id + 2] = length
    append(`${id + 2} 0 obj\n<< /Type /XObject /Subtype /Image /Width ${image.width} /Height ${image.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${image.bytes.byteLength} >>\nstream\n`)
    appendBytes(image.bytes)
    append('\nendstream\nendobj\n')
  })
  const xref = length
  append(`xref\n0 ${offsets.length}\n0000000000 65535 f \n`)
  offsets.slice(1).forEach(offset => append(`${String(offset).padStart(10, '0')} 00000 n \n`))
  append(`trailer\n<< /Size ${offsets.length} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`)
  return new Blob(parts, { type: 'application/pdf' })
}

/** Creates a downloadable, raster-sharp A4 landscape reference PDF. The caller owns download/UI. */
export async function exportVendorSetupPdf(input: VendorSetupPdfInput): Promise<{ filename: string; blob: Blob; pageCount: number }> {
  const { data, draft, language, view, viewMode = 'folded', additionalView } = input
  const t = COPY[language]
  const rtl = language === 'he'
  const allMarks = new Map(draft.marks.map(mark => [mark.feature_id, mark]))
  const orderedIds = [...new Set(draft.sequence)]
  const ordered = orderedIds.map(id => allMarks.get(id)).filter((mark): mark is VendorBendMark => !!mark && mark.status === 'active')
  const active = [...ordered, ...draft.marks.filter(mark => mark.status === 'active' && !orderedIds.includes(mark.feature_id))]
  const measurements = draft.general_measurements ?? []
  if (active.length > MAX_ROWS || measurements.length > MAX_ROWS || (draft.machine_moves?.length ?? 0) > MAX_ROWS) {
    throw new Error('PDF export supports at most 200 bends, moves and reference measurements per draft.')
  }
  const authoredMoves = draft.machine_moves_explicit === true ? draft.machine_moves ?? [] : []
  const moveById = new Map<string, number>()
  const groups: Array<{ title: string; ids: string[] }> = []
  authoredMoves.forEach((move, index) => {
    const ids = move.filter(id => allMarks.get(id)?.status === 'active' && !moveById.has(id))
    if (!ids.length) return
    ids.forEach(id => moveById.set(id, index + 1))
    groups.push({ title: `${t.move} ${index + 1}`, ids })
  })
  const ungrouped = active.filter(mark => !moveById.has(mark.feature_id)).map(mark => mark.feature_id)
  if (ungrouped.length) groups.push({ title: t.ungrouped, ids: ungrouped })
  const colorById = new Map(draft.marks.map((mark, index) => {
    const colorIndex = moveById.has(mark.feature_id) ? moveById.get(mark.feature_id)! - 1 : authoredMoves.length + index
    return [mark.feature_id, COLORS[colorIndex] || `hsl(${Math.round((colorIndex * 137.508) % 360)} 95% 68%)`]
  }))
  const flangeById = new Map(data.flange_surfaces?.map(surface => [surface.feature_id, surface]) ?? [])
  await document.fonts.ready
  await Promise.all([document.fonts.load(`400 21px ${FONT}`), document.fonts.load(`700 21px ${FONT}`)])
  const [image, additionalImage] = await Promise.all([decodeView(view), decodeView(additionalView?.view ?? null)])
  const pages: PageImage[] = []
  const annotationRows: Cell[][] = []
  const exportedAt = new Date()
  const timestamp = exportedAt.toLocaleString(language === 'he' ? 'he-IL' : 'en-GB', { hour12: false })

  const newSheet = (section: string): Sheet => {
    const canvas = document.createElement('canvas')
    canvas.width = PIXEL_WIDTH
    canvas.height = PIXEL_HEIGHT
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas rendering is not available for PDF export.')
    ctx.scale(PIXEL_WIDTH / WIDTH, PIXEL_HEIGHT / HEIGHT)
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, WIDTH, HEIGHT)
    box(ctx, 20, 20, WIDTH - 40, HEIGHT - 40)
    box(ctx, MARGIN, 36, WIDTH - 2 * MARGIN, 145)
    text(ctx, 'SHERMAN QC / VENDOR SETUP', MARGIN + 18, 47, 570, { size: 17, bold: true, color: MUTED })
    text(ctx, t.title, MARGIN + 18, 77, WIDTH - 2 * MARGIN - 36, { rtl, size: 35, bold: true })
    const meta = `${t.part}: ${cleanText(data.part.part_number)}   |   ${t.revision}: ${cleanText(data.part.revision) || '-'}   |   ${t.material}: ${cleanText(data.part.material) || '-'}`
    text(ctx, meta, MARGIN + 18, 126, WIDTH - 2 * MARGIN - 36, { rtl, size: 20 })
    text(ctx, cleanText(data.part.part_name), MARGIN + 18, 158, WIDTH - 2 * MARGIN - 36, { rtl, size: 16, color: MUTED })
    text(ctx, section, MARGIN, CONTENT_TOP, WIDTH - 2 * MARGIN, { rtl, size: 25, bold: true })
    box(ctx, MARGIN, 1018, WIDTH - 2 * MARGIN, 132, '#f6f8fa')
    text(ctx, t.reference, MARGIN + 14, 1027, WIDTH - 2 * MARGIN - 28, { rtl, size: 18, bold: true })
    text(ctx, `${t.draftDisclosure} ${t.status}: ${draft.operator_sequence_confirmed ? t.confirmed : t.pending}.`, MARGIN + 14, 1053, WIDTH - 2 * MARGIN - 28, { rtl, size: 16 })
    paragraph(ctx, t.units, MARGIN + 14, 1075, WIDTH - 2 * MARGIN - 28, { rtl, size: 16, lineHeight: 21 })
    paragraph(ctx, t.safety, MARGIN + 14, 1101, WIDTH - 2 * MARGIN - 28, { rtl, size: 16, lineHeight: 21 })
    text(ctx, `${t.issued}: ${timestamp}`, MARGIN, 1157, 1050, { size: 15, color: MUTED })
    text(ctx, `${t.sheet} ${pages.length + 1}`, WIDTH - 220, 1155, 184, { rtl, size: 17, bold: true })
    return { canvas, ctx }
  }
  const finish = async (sheet: Sheet) => {
    pages.push(await jpeg(sheet))
    // Release the large drawing surface before composing the next sheet.
    sheet.canvas.width = 1
    sheet.canvas.height = 1
  }

  const drawing = { x: MARGIN, y: 246, width: 1070, height: 674 }
  const legendX = 1130
  const legendWidth = WIDTH - MARGIN - legendX

  const legendRows = groups.flatMap(group => [
    { title: group.title, groupTitle: group.title, mark: null },
    ...group.ids.map(id => ({ title: '', groupTitle: group.title, mark: allMarks.get(id)! })),
  ])
  let legendIndex = 0
  const drawLegend = (sheet: Sheet, x: number, y: number, width: number, bottom: number) => {
    const startY = y
    if (legendRows[legendIndex]?.mark) {
      box(sheet.ctx, x, y, width, 38, '#f0f3f6')
      text(sheet.ctx, `${legendRows[legendIndex].groupTitle} (${t.sectionContinued})`, x + 12, y + 7, width - 24, { rtl, size: 20, bold: true })
      y += 38
    }
    while (legendIndex < legendRows.length) {
      const row = legendRows[legendIndex]
      const value = row.mark ? `${row.mark.label}  |  ${row.mark.review_status === 'confirmed' ? t.reviewed : t.unreviewed}` : row.title
      const lines = wrap(sheet.ctx, value, width - 58, 20, !row.mark)
      const rowHeight = Math.max(36, lines.length * 28 + 10)
      const next = legendRows[legendIndex + 1]
      const nextHeight = !row.mark && next?.mark ? Math.max(36, wrap(sheet.ctx, `${next.mark.label}  |  ${next.mark.review_status === 'confirmed' ? t.reviewed : t.unreviewed}`, width - 58, 20).length * 28 + 10) : 0
      if (y + rowHeight + nextHeight > bottom) {
        if (bottom - startY > 650 && y - startY < 45) throw new Error('A bend label is too long for the PDF legend. Shorten the label before export.')
        break
      }
      if (!row.mark) box(sheet.ctx, x, y, width, rowHeight, '#f0f3f6')
      else {
        sheet.ctx.fillStyle = colorById.get(row.mark.feature_id) || INK
        sheet.ctx.fillRect(rtl ? x + width - 25 : x + 12, y + 11, 12, 12)
      }
      paragraph(sheet.ctx, value, x + (rtl ? 12 : 35), y + 7, width - 58, { rtl, size: 20, bold: !row.mark, lineHeight: 28 })
      y += rowHeight
      legendIndex++
    }
    if (legendIndex < legendRows.length) text(sheet.ctx, `${t.sectionContinued}...`, x + 12, bottom + 9, width - 24, { rtl, size: 17, color: MUTED })
    return y
  }
  const summary = `${t.active}: ${active.length}   |   ${t.moves}: ${groups.filter(group => group.title !== t.ungrouped).length}   |   ${t.status}: ${draft.operator_sequence_confirmed ? t.confirmed : t.pending}`
  const provenance = `${t.draftRevision}: ${draft.draft_revision ?? draft.expected_draft_revision}   |   ${t.hash}: ${cleanText(draft.cad_file_hash) || '-'}`
  const drawViewSheet = async (capturedView: VendorSetupPdfView | null, capturedImage: HTMLImageElement | null, mode: VendorSetupPdfViewMode) => {
    const title = mode === 'unfolded_visual' ? t.unfoldedView : t.foldedView
    const sheet = newSheet(title)
    const ctx = sheet.ctx
    box(ctx, drawing.x, drawing.y, drawing.width, drawing.height)
    const disclaimerHeight = mode === 'unfolded_visual'
      ? paragraph(ctx, t.unfoldedDisclaimer, drawing.x + 16, drawing.y + 13, drawing.width - 32, { rtl, size: 17, bold: true, lineHeight: 24 }) + 20
      : 0
    const inner = { x: drawing.x + 16, y: drawing.y + 16 + disclaimerHeight, width: drawing.width - 32, height: drawing.height - 65 - disclaimerHeight }
    if (capturedImage && capturedView) {
      const labels = capturedView.labels.map((label, index) => ({ ...label, index, labelText: cleanText(label.text) }))
      const railWidth = labels.length && labels.length <= 24 ? 210 : 0
      const imageArea = { x: inner.x + railWidth + (railWidth ? 12 : 0), y: inner.y, width: inner.width - 2 * railWidth - (railWidth ? 24 : 0), height: inner.height }
      const fit = Math.min(imageArea.width / capturedView.width, imageArea.height / capturedView.height)
      const imageWidth = capturedView.width * fit
      const imageHeight = capturedView.height * fit
      const imageX = imageArea.x + (imageArea.width - imageWidth) / 2
      const imageY = imageArea.y + (imageArea.height - imageHeight) / 2
      ctx.drawImage(capturedImage, imageX, imageY, imageWidth, imageHeight)
      const anchor = (label: typeof labels[number]) => Number.isFinite(label.x) && Number.isFinite(label.y) && label.x >= 0 && label.y >= 0 && label.x <= capturedView.width && label.y <= capturedView.height
        ? { x: imageX + label.x * fit, y: imageY + label.y * fit } : null
      const labelColor = (label: typeof labels[number]) => {
        const authored = label.featureId ? colorById.get(label.featureId) : undefined
        return authored || (CSS.supports('color', label.color) ? label.color : INK)
      }
      const addKey = (label: typeof labels[number]) => annotationRows.push([{ text: `${title} / #${label.index + 1}`, color: labelColor(label) }, { text: label.labelText || '-', ltr: true }])
      if (railWidth) {
        // Fixed, balanced rails keep dense bend labels separate without moving their geometric anchors.
        const sorted = [...labels].sort((a, b) => (Number.isFinite(a.x) ? a.x : 0) - (Number.isFinite(b.x) ? b.x : 0) || a.index - b.index)
        const middle = Math.ceil(sorted.length / 2)
        const rails = [sorted.slice(0, middle), sorted.slice(middle)]
        rails.forEach((rail, side) => {
          rail.sort((a, b) => (Number.isFinite(a.y) ? a.y : 0) - (Number.isFinite(b.y) ? b.y : 0) || a.index - b.index)
          const slotHeight = inner.height / rail.length
          rail.forEach((label, row) => {
            const fullLines = wrap(ctx, label.labelText || '-', railWidth - 24, 18, true)
            const fits = fullLines.length * 24 + 12 <= slotHeight - 8
            const lines = fits ? fullLines : [`#${label.index + 1}`]
            if (!fits || !anchor(label)) addKey(label)
            const height = lines.length * 24 + 12
            const x = side ? inner.x + inner.width - railWidth : inner.x
            const y = inner.y + row * slotHeight + (slotHeight - height) / 2
            const target = anchor(label)
            if (target) {
              const edgeX = side ? x : x + railWidth
              ctx.strokeStyle = labelColor(label)
              ctx.lineWidth = 1.6
              ctx.beginPath()
              ctx.moveTo(target.x, target.y)
              ctx.lineTo(side ? edgeX - 10 : edgeX + 10, y + height / 2)
              ctx.lineTo(edgeX, y + height / 2)
              ctx.stroke()
              ctx.fillStyle = labelColor(label)
              ctx.beginPath()
              ctx.arc(target.x, target.y, 3.5, 0, Math.PI * 2)
              ctx.fill()
            }
            box(ctx, x, y, railWidth, height, '#ffffff')
            ctx.fillStyle = labelColor(label)
            ctx.fillRect(x, y, 5, height)
            lines.forEach((line, lineIndex) => text(ctx, line, x + 12, y + 6 + lineIndex * 24, railWidth - 24, { size: 18, bold: true }))
          })
        })
      } else {
        labels.forEach(label => {
          addKey(label)
          const target = anchor(label)
          if (!target) return
          const x = Math.max(inner.x, Math.min(inner.x + inner.width - 36, target.x - 18))
          const y = Math.max(inner.y, Math.min(inner.y + inner.height - 26, target.y - 13))
          box(ctx, x, y, 36, 26, '#ffffff')
          text(ctx, String(label.index + 1), x + 3, y + 2, 30, { size: 17, bold: true, color: labelColor(label) })
        })
      }
    } else paragraph(ctx, t.noView, drawing.x + 35, inner.y + inner.height / 2 - 30, drawing.width - 70, { rtl, size: 25 })
    text(ctx, `${title} / ${t.notToScale}`, drawing.x + 14, drawing.y + drawing.height - 36, drawing.width - 28, { rtl, size: 17, bold: true })
    box(ctx, legendX, 246, legendWidth, 674)
    text(ctx, t.legend, legendX + 16, 262, legendWidth - 32, { rtl, size: 22, bold: true })
    legendIndex = 0
    if (legendRows.length) drawLegend(sheet, legendX + 16, 310, legendWidth - 32, 887)
    else paragraph(ctx, t.noGroups, legendX + 16, 320, legendWidth - 32, { rtl, size: 21 })
    paragraph(ctx, summary, MARGIN, 938, WIDTH - 2 * MARGIN, { rtl, size: 20, bold: true, lineHeight: 27 })
    text(ctx, t.foldedReferences, MARGIN, 968, WIDTH - 2 * MARGIN, { rtl, size: 16, bold: true, color: MUTED })
    text(ctx, provenance, MARGIN, 991, WIDTH - 2 * MARGIN, { size: 14, color: MUTED })
    await finish(sheet)
  }
  await drawViewSheet(view, image, viewMode)
  // A missing optional capture is never replaced with a blank or misleading unfolded sheet.
  if (additionalView?.view && additionalImage) await drawViewSheet(additionalView.view, additionalImage, additionalView.mode)
  while (legendIndex < legendRows.length) {
    const sheet = newSheet(t.legendContinuation)
    const columnWidth = (WIDTH - 2 * MARGIN - 36) / 3
    for (let column = 0; column < 3 && legendIndex < legendRows.length; column++) {
      const x = rtl ? WIDTH - MARGIN - columnWidth - column * (columnWidth + 18) : MARGIN + column * (columnWidth + 18)
      drawLegend(sheet, x, 250, columnWidth, CONTENT_BOTTOM - 40)
    }
    await finish(sheet)
  }

  const drawTable = async (title: string, headers: Cell[], widths: number[], rows: Cell[][], explanation: string) => {
    let index = 0
    let continued = false
    do {
      const sheet = newSheet(`${title}${continued ? ` - ${t.sectionContinued}` : ''}`)
      let y = 246
      y += paragraph(sheet.ctx, explanation, MARGIN, y, WIDTH - 2 * MARGIN, { rtl, size: 18, lineHeight: 25 }) + 20
      const headerLines = headers.map((cell, column) => wrap(sheet.ctx, cell.text, widths[column] - 20, 19, true))
      const headerHeight = Math.max(...headerLines.map(lines => lines.length)) * 27 + 20
      let x = MARGIN
      headers.forEach((cell, column) => {
        const cellX = rtl ? WIDTH - MARGIN - (x - MARGIN) - widths[column] : x
        box(sheet.ctx, cellX, y, widths[column], headerHeight, '#eaf0f5')
        paragraph(sheet.ctx, cell.text, cellX + 10, y + 10, widths[column] - 20, { rtl: rtl && !cell.ltr, size: 19, bold: true, lineHeight: 27 })
        x += widths[column]
      })
      y += headerHeight
      if (!rows.length) paragraph(sheet.ctx, title === t.bendTable ? t.noBends : t.noMeasurements, MARGIN + 12, y + 24, WIDTH - 2 * MARGIN - 24, { rtl })
      while (index < rows.length) {
        const row = rows[index]
        const lines = row.map((cell, column) => wrap(sheet.ctx, cell.text, widths[column] - 20, 20))
        const rowHeight = Math.max(...lines.map(value => value.length)) * 28 + 22
        if (rowHeight > CONTENT_BOTTOM - y && y > 246 + headerHeight + 80) break
        if (rowHeight > CONTENT_BOTTOM - y) throw new Error('An annotation is too long for a PDF table row. Shorten the label or measurement note before export.')
        x = MARGIN
        row.forEach((cell, column) => {
          const cellX = rtl ? WIDTH - MARGIN - (x - MARGIN) - widths[column] : x
          box(sheet.ctx, cellX, y, widths[column], rowHeight, index % 2 ? '#f7f9fb' : '#ffffff')
          if (cell.color) { sheet.ctx.fillStyle = cell.color; sheet.ctx.fillRect(cellX + 1, y + 1, 5, rowHeight - 2) }
          lines[column].forEach((line, lineIndex) => text(sheet.ctx, line, cellX + 10, y + 11 + lineIndex * 28, widths[column] - 20, { rtl: rtl && !cell.ltr, size: 20 }))
          x += widths[column]
        })
        y += rowHeight
        index++
      }
      await finish(sheet)
      continued = true
    } while (index < rows.length)
  }

  if (annotationRows.length) await drawTable(t.annotationKey, [{ text: t.annotation }, { text: t.annotationDefinition }], [572, 1040], annotationRows, t.annotationHelp)

  const bendRows: Cell[][] = active.map(mark => {
    const flange = flangeById.get(mark.feature_id)
    const depth = flange?.depth_status === 'UNAVAILABLE_AMBIGUOUS_BOUNDARY' ? null : flange?.nominal_flange_depth_mm
    return [
      { text: draft.operator_sequence_confirmed && orderedIds.includes(mark.feature_id) ? String(orderedIds.indexOf(mark.feature_id) + 1) : '-', ltr: true },
      { text: mark.label, ltr: true, color: colorById.get(mark.feature_id) },
      { text: moveById.has(mark.feature_id) ? String(moveById.get(mark.feature_id)) : '-', ltr: true },
      { text: number(mark.target_angle), ltr: true },
      { text: number(distance(mark.line_start, mark.line_end)), ltr: true },
      { text: number(depth), ltr: true },
      { text: `${mark.source === 'manual' ? t.manual : t.cad} / ${mark.review_status === 'confirmed' ? t.reviewed : t.unreviewed}` },
    ]
  })
  await drawTable(t.bendTable, [{ text: t.order }, { text: t.bend }, { text: t.move }, { text: t.angle }, { text: t.width }, { text: t.depth }, { text: t.sourceReview }], [90, 170, 100, 290, 270, 270, 422], bendRows, `${t.definitions}\n${t.status}: ${draft.operator_sequence_confirmed ? t.confirmed : t.pending}`)

  const measurementRows: Cell[][] = measurements.map((measurement, index) => {
    const bends = measurement.bend_ids.map(id => allMarks.get(id)?.label || id).join(' / ')
    return [
      { text: `R${index + 1} · ${measurement.label}` },
      { text: `${measurement.kind === 'bend_pair' ? `${t.pair}: ${bends}` : t.points}\nA: ${point(measurement.point_start)}\nB: ${point(measurement.point_end)}` },
      { text: number(distance(measurement.point_start, measurement.point_end)), ltr: true },
      { text: number(measurement.target_mm), ltr: true },
      { text: measurement.note || '-' },
    ]
  })
  await drawTable(t.measurements, [{ text: t.generalLabel }, { text: t.definition }, { text: t.distance }, { text: t.target }, { text: t.note }], [200, 572, 250, 220, 370], measurementRows, t.generalDefinitions)

  const notes = [
    ...(data.part.notes ? [{ title: t.partNotes, body: data.part.notes }] : []),
    ...draft.marks.filter(mark => mark.note).map(mark => ({ title: `${mark.label}${mark.status === 'removed' ? ` (${t.removedNotes})` : ''}`, body: mark.note })),
  ]
  let noteIndex = 0
  let remainingLines: string[] = []
  let continuedNotes = false
  while (noteIndex < notes.length) {
    const sheet = newSheet(`${t.notes}${continuedNotes ? ` - ${t.sectionContinued}` : ''}`)
    let y = 256
    while (noteIndex < notes.length) {
      if (!remainingLines.length) remainingLines = wrap(sheet.ctx, `${notes[noteIndex].title}\n${notes[noteIndex].body}`, WIDTH - 2 * MARGIN - 32, 21)
      const available = Math.floor((CONTENT_BOTTOM - y - 20) / 30)
      if (available < 2) break
      const taken = remainingLines.splice(0, available)
      const height = taken.length * 30 + 20
      box(sheet.ctx, MARGIN, y, WIDTH - 2 * MARGIN, height)
      taken.forEach((line, lineIndex) => text(sheet.ctx, line, MARGIN + 16, y + 10 + lineIndex * 30, WIDTH - 2 * MARGIN - 32, { rtl, size: 21 }))
      y += height + 18
      if (!remainingLines.length) noteIndex++
      else break
    }
    await finish(sheet)
    continuedNotes = true
  }

  return {
    filename: `${filenamePart(data.part.part_number)}-vendor-setup-${exportedAt.toISOString().slice(0, 10)}.pdf`,
    blob: pdf(pages),
    pageCount: pages.length,
  }
}
