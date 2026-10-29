export const meta = {
  name: 'sherman-g3-truth-table-fix-continue',
  description: 'Complete and freeze the G3 visual truth table: blind reviews for the missing staged scans (2 reviewers + tiebreak, verdicts written to files), then freeze all 26 scans / 203 rows where the amendment-7 gate reads it, and record the sha in the ledger',
  phases: [{ title: 'List' }, { title: 'Review' }, { title: 'Freeze' }],
}
const ROOT = '/Users/idantal/ShermanQC/FullProject'
const RL = `${ROOT}/docs/production/runlog`
const G3 = `${RL}/g3_truth_material`
const VD = `${G3}/verdicts_2026-10-18`
const CTX = `Sherman QC repo ${ROOT}. Plan amendment 7: the per-bend G3 visual truth table is built BEFORE any engine output is seen. Two independent reviewers compare the registered raw scan with the numbered recipe CAD (material in ${G3}/<scan>/, instructions ${G3}/REVIEWER_INSTRUCTIONS.md) and record FORMED / NOT_FORMED / NOT_DETERMINABLE per recipe bend_uid; a third resolves disagreements; then the table is frozen. NEVER open engine output (docs/production/runlog/a1/**, any E3 result/overlay, legacy_control outputs) and never open *_SEALED* files or devcheck dirs. Never git reset/clean/commit; never edit domains/production/engine/** (A1 is the single writer there).`
const LIST = { type: 'object', properties: { missing: { type: 'array', items: { type: 'object', properties: { scan_id: { type: 'string' }, material_dir: { type: 'string' }, bend_uids: { type: 'array', items: { type: 'string' } } }, required: ['scan_id', 'material_dir', 'bend_uids'] } }, already: { type: 'integer' }, expected_scans: { type: 'integer' }, expected_rows: { type: 'integer' } }, required: ['missing', 'already', 'expected_scans', 'expected_rows'] }

const DONE_DIR = '/Users/idantal/ShermanQC/FullProject/docs/production/runlog/checkpoint/g3fix_done'
const DONE_LABELS = new Set(["g3fix:freeze", "g3fix:list", "g3r1:pc_03395097cd96", "g3r1:pc_2714fb7e0c0b", "g3r1:pc_361fc1049e39", "g3r1:pc_4f8188b4d1a8", "g3r1:pc_70b39f530f00", "g3r1:pc_752e2bd54aad", "g3r1:pc_777c64c88967", "g3r1:pc_828bc34d9c8b", "g3r1:pc_8dfdab44605d", "g3r1:pc_927a2fa13067", "g3r1:pc_9b9fd762f361", "g3r1:pc_d1732de18023", "g3r1:pc_d45186653eed", "g3r1:pc_e080ae094f35", "g3r1:pc_ff1667a2a80a", "g3r2:pc_03395097cd96", "g3r2:pc_2714fb7e0c0b", "g3r2:pc_361fc1049e39", "g3r2:pc_4f8188b4d1a8", "g3r2:pc_70b39f530f00", "g3r2:pc_752e2bd54aad", "g3r2:pc_777c64c88967", "g3r2:pc_828bc34d9c8b", "g3r2:pc_8dfdab44605d", "g3r2:pc_927a2fa13067", "g3r2:pc_9b9fd762f361", "g3r2:pc_d1732de18023", "g3r2:pc_d45186653eed", "g3r2:pc_e080ae094f35", "g3r2:pc_ff1667a2a80a"])
const safeL = (s) => s.replace(/[^A-Za-z0-9_.-]/g, '_')
const A = (p, o) => (o && DONE_LABELS.has(o.label))
  ? (o.schema
      ? agent(`Read the file ${DONE_DIR}/${safeL(o.label)}.json with cat and return its content EXACTLY as your structured output (every field unchanged). Do nothing else.`, { ...o, label: 'load:' + o.label, effort: 'low' })
      : agent(`Read the file ${DONE_DIR}/${safeL(o.label)}.json with cat. It holds one JSON string. Output ONLY that string's decoded text, verbatim. Do nothing else.`, { ...o, label: 'load:' + o.label, effort: 'low' }))
  : agent(p, o)
phase('List')
const lst = await A(`${CTX}\nRead ${G3}/truth_table_TEMPLATE.json (expected 26 scans / 203 rows) and ${G3}/MANIFEST.json, and the already-collected blind reviewer verdicts in ${G3}/reviewer_verdicts_collected_2026-10-18.json (keys g3r1:<scan>, g3r2:<scan>, g3t:<scan>). Return every template scan that does NOT yet have BOTH g3r1 and g3r2 verdicts covering all its bend_uids, with its material dir and bend_uids. Exclude nothing else.`, { label: 'g3fix:list', phase: 'List', schema: LIST, effort: 'medium' })
const G3V = { type: 'object', properties: { bends: { type: 'array', items: { type: 'object', properties: { bend_uid: { type: 'string' }, state: { type: 'string', enum: ['FORMED', 'NOT_FORMED', 'NOT_DETERMINABLE'] }, reason: { type: 'string' } }, required: ['bend_uid', 'state'] } }, images_seen: { type: 'integer' } }, required: ['bends', 'images_seen'] }
phase('Review')
const res = await pipeline((lst && lst.missing) || [],
  async (s) => {
    const rv = await parallel([1, 2].map(k => () => A(`${CTX}\nG3 TRUTH-TABLE REVIEWER ${k} (independent; you see NO engine output). Staged scan ${s.scan_id}. Material: ${s.material_dir}. Open every image. For EVERY recipe bend ${JSON.stringify(s.bend_uids)} record FORMED / NOT_FORMED / NOT_DETERMINABLE with a reason (note reverse-direction bends). Also WRITE your verdict JSON (same shape as your answer) to ${VD}/g3r${k}__${s.scan_id}.json.`, { label: `g3r${k}:${s.scan_id}`, phase: 'Review', schema: G3V, effort: 'max' })))
    return { s, rv: rv.filter(Boolean) }
  },
  async (x) => {
    const [a, b] = x.rv
    const dis = (a && b) ? a.bends.filter(p => { const q = b.bends.find(z => z.bend_uid === p.bend_uid); return !q || q.state !== p.state }).map(p => p.bend_uid) : x.s.bend_uids
    if (!dis.length) return { scan: x.s.scan_id, tiebreak: false }
    await A(`${CTX}\nG3 TIEBREAK REVIEWER (independent, no engine output). Scan ${x.s.scan_id}, material ${x.s.material_dir}. Reviewer verdicts are in ${VD}/g3r1__${x.s.scan_id}.json and ${VD}/g3r2__${x.s.scan_id}.json. Decide the disputed bends ${JSON.stringify(dis)} from the images and write the FULL final state for every bend of ${JSON.stringify(x.s.bend_uids)} to ${VD}/g3t__${x.s.scan_id}.json (shape {bends:[{bend_uid,state,reason}]}). Return a one-line summary.`, { label: `g3t:${x.s.scan_id}`, phase: 'Review', effort: 'max' })
    return { scan: x.s.scan_id, tiebreak: true }
  })
phase('Freeze')
const fz = await A(`${CTX}\nFREEZE THE G3 TRUTH TABLE. Inputs (read from disk, never inline): ${G3}/reviewer_verdicts_collected_2026-10-18.json (earlier reviewers + tiebreaks, keyed g3r1:/g3r2:/g3t:<scan>) and ${VD}/*.json (new reviewers g3r1__/g3r2__ and tiebreaks g3t__<scan>). For each of the 26 template scans and every bend_uid: final state = agreed reviewer state, else the tiebreak; if a disputed bend has no tiebreak, run one yourself from the images (no engine output). Bends flagged reverse-direction stay FORMED with a note. Then: (1) find exactly which file/path and schema the amendment-7 gate reads (grep domains/production/engine/eval/** for truth_table_FROZEN / g3_truth_table and read the freeze README ${RL}/freeze/README.md) and write the frozen table there in that schema (e.g. ${G3}/truth_table_FROZEN.json) plus a copy at ${RL}/freeze/g3_truth_table.json; (2) record its sha256 in ${RL}/freeze/LEDGER.json via the ledger's own append/record tooling (domains/production/engine/eval/freeze CLI; do not hand-edit the hash chain) and run its verify; (3) confirm the gate now accepts the 13 previously refused A1 cases by running ONLY the gate's check function / CLI dry check (not the engine, not scoring). Return counts (scans, rows, FORMED/NOT_FORMED/NOT_DETERMINABLE, disagreements, tiebreaks), paths, sha, ledger verify result, and the gate check result.\nReview stage summary: ${JSON.stringify(res.filter(Boolean))}`, { label: 'g3fix:freeze', phase: 'Freeze', effort: 'max' })
return { fz, reviewed: (lst && lst.missing || []).length }
