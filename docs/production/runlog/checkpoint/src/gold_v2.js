export const meta = {
  name: 'sherman-gold-set-v2',
  description: 'Extend the vision gold set: all remaining CONSUMED real scans (no-CAD for the blind benchmark + the few remaining CAD scans), prep where missing, 3 blind auditors, fix + 2 blind verifiers, write gold labels, build GOLD_SET_V2 (V1 + new) with CAD/no-CAD split, rescore existing lanes',
  phases: [
    { title: 'Select' }, { title: 'Prep' }, { title: 'Audit' }, { title: 'Fix' }, { title: 'Write' },
  ],
}
const ROOT = '/Users/idantal/ShermanQC/FullProject'
const RL = `${ROOT}/docs/production/runlog`
const OUT = `${RL}/gold_set_v2`
const LOCK = `until mkdir /private/tmp/claude-501/sherman_geom.lock 2>/dev/null; do sleep 3; done; <cmd>; rc=$?; rmdir /private/tmp/claude-501/sherman_geom.lock`
const CTX = `
Context: Sherman QC repo ${ROOT}. Owner request (2026-10-18): more gold-labelled scans so the NO-CAD (blind) track can be benchmarked broadly; production will be CAD-assisted, so also add any remaining CAD scans. Gold labels are marked/checked by Claude with HEAVY VISION: every line verified by eye on renders, close-ups and perpendicular cross-sections. GOLD_SET_V1 (${RL}/gold_set_v1/GOLD_SET_V1.json, docs/production/GOLD_SET_V1.md; 58 scans) already exists — do NOT redo any scan in it.
Gold rules: LANDS iff the line lies on the OUTER-skin apex of a real formed fold, parallel, covering >= 80% of the visible fold. Joggle = 2 bends; formed bends >= 10 deg count (owner rule 2026-10-18; was 15); a fold interrupted by windows/reliefs folding the same two plates = ONE line; hole walls, thickness/perimeter edges, embosses, gentle rolled kinks, artifacts are not bends. Staged scans: label only bends formed in that scan. Zero-bend scans are valid gold with no lines.
Tools: ${ROOT}/.venv/bin/python -m research.bench_labeling {prep,lines,write-label} (README_OVERRIDES.md: margin_mm, extent_t, plane_from_points, apex_mode station_median). Heavy commands (prep, lines, full-scan loads) only under the lock: ${LOCK}; OMP_NUM_THREADS=1; check rc yourself. Light renders from points_ds.npz (matplotlib Agg) need no lock.
Rules: never git reset/clean/commit; never modify human_bend_lines_v1, ai_bend_lines_v1 or ai_bend_lines_gold11_relabel_v1; gold labels go to data/bend_corpus/annotations/ai_bend_lines_gold_v1; don't touch domains/production, contracts/production, apps, frontend. Custody: CONSUMED scans only (docs/production/runlog/benchmark_v3/BENCHMARK_V3.json). FORBIDDEN: FRESH scans, *_SEALED*, incoming_codex_fetch FRESH files, HOLDOUT10, sealed truth, N001-N016, EHO4, LIDAR_DEMO synthetic scans, EXCLUDED_NOT_A_SCAN / CAD reference meshes, the owner's face scans. Reuse existing prep/lines/renders.`

phase('Select')
const CAND = { type: 'object', properties: { scans: { type: 'array', items: { type: 'object', properties: {
  scan_id: { type: 'string' }, part_id: { type: 'string' }, family: { type: 'string' }, stage: { type: 'string' }, path: { type: 'string' },
  lines_dir: { type: 'string', description: 'existing lines dir or empty' }, prep_dir: { type: 'string', description: 'existing prep dir (render/ + points_ds.npz) or empty' },
  n_lines: { type: 'integer' }, has_step: { type: 'boolean' }, cad_bend_lines: { type: 'integer' }, current_truth: { type: 'string' } },
  required: ['scan_id', 'part_id', 'family', 'stage', 'path', 'lines_dir', 'prep_dir', 'n_lines', 'has_step', 'current_truth'] } }, note: { type: 'string' } }, required: ['scans', 'note'] }
const sel = await agent(`${CTX}\nTASK: SELECT every remaining CONSUMED real scan for gold: all CONSUMED benchmark_scan rows in BENCHMARK_V3.json NOT already in GOLD_SET_V1 (match by scan_id AND by same-mesh twins: if a PLY/STL twin is gold, skip the other), whose truth is AI_GOLD_ACCEPTED, GOLD_LINES_AI_LEGACY, GOLD_LINES_HUMAN, CAD_RECIPE or NONE — CAD_RECIPE/NONE scans need labelling from scratch. Prefer PLY over STL/OBJ twins (write-label only accepts PLY; if only STL/OBJ exists, include it only if a same-frame PLY can be produced by prep — else list it in note as skipped). No per-family cap this time (the blind benchmark wants breadth), but skip exact duplicates. Include GOLD_SET_V1's reserves (sc_2852c2845b94 etc.) and sc_186463746daa is OUT (already decided). sc_90257b128d80 stays out (possibly FRESH). For each give existing lines_dir/prep_dir (benchmark_v2/label_lines*, label_lines_r2/r3, benchmark_v3/label_lines, label_prep) or empty strings. Write ${OUT}/candidates.json. Return the list and a note of everything skipped with reasons.`, { label: 'select', phase: 'Select', schema: CAND, effort: 'high' })
const cands = (sel && sel.scans) || []
log(`gold v2 candidates: ${cands.length} (no-CAD ${cands.filter(c => !c.has_step).length}, CAD ${cands.filter(c => c.has_step).length})`)

const AUD = { type: 'object', properties: {
  lines: { type: 'array', items: { type: 'object', properties: { label: { type: 'string' }, verdict: { type: 'string', enum: ['OK', 'OFFSET', 'WRONG_EXTENT', 'NOT_A_BEND', 'UNCLEAR'] }, offset_mm: { type: 'number' }, coverage_pct: { type: 'number' }, reason: { type: 'string' } }, required: ['label', 'verdict'] } },
  missing_bends: { type: 'array', items: { type: 'string' } }, real_bend_count: { type: 'integer' }, images_seen: { type: 'integer' } },
  required: ['lines', 'missing_bends', 'real_bend_count', 'images_seen'] }
const VV = { type: 'object', properties: { pass: { type: 'boolean' }, images_seen: { type: 'integer' }, problems: { type: 'string' } }, required: ['pass', 'images_seen'] }
const clean = (a) => a && a.images_seen > 0 && a.missing_bends.length === 0 && a.lines.every(l => l.verdict === 'OK')

const res = await pipeline(cands,
  async (c) => {
    if (c.prep_dir && c.lines_dir) return { c }
    const r = await agent(`${CTX}\nPREP/INITIAL LABEL for scan ${c.scan_id} (part ${c.part_id}, stage '${c.stage}'). ${c.prep_dir ? `Prep exists at ${c.prep_dir}.` : `Run prep for ${c.path} into ${OUT}/prep/${c.scan_id} under the lock (retry once with --voxel 1.0 --max-points 150000 on failure).`} ${c.lines_dir ? `Lines exist at ${c.lines_dir}.` : `There is no usable line set: open the numbered patch renders, write pairs.json for every real formed bend (or no_bends_present) into ${OUT}/lines/${c.scan_id}, run lines under the lock, and check every line yourself with close-ups/sections.`} Return EXACTLY two lines: 'PREP <dir>' and 'LINES <dir>' (or 'GIVE_UP <reason>').`, { label: `prep:${c.scan_id}`, phase: 'Prep', effort: 'high' })
    const s = String(r)
    if (/GIVE_UP/.test(s)) return { c, giveup: s.slice(0, 600) }
    const pm = s.match(/PREP\s+(\S+)/), lm = s.match(/LINES\s+(\S+)/)
    return { c: { ...c, prep_dir: pm ? pm[1] : c.prep_dir, lines_dir: lm ? lm[1] : c.lines_dir } }
  },
  async (x) => {
    if (x.giveup) return { ...x, final: 'EXCLUDED', reason: x.giveup }
    const c = x.c
    const ap = (k) => `${CTX}\nINDEPENDENT GOLD AUDITOR ${k} of 3 for scan ${c.scan_id} (part ${c.part_id}, stage '${c.stage}', CAD bend lines ${c.cad_bend_lines ?? -1}). Current lines: ${c.lines_dir} (lines.json + render/). Prep: ${c.prep_dir}. Open every render; then make your OWN renders from points_ds.npz with lines.json overlaid into /private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/goldv2/${c.scan_id}/a${k}/: >= 8 global views, a close-up per line, perpendicular sections at 25/50/75%. Judge every line (OK / OFFSET (> 1 mm off the outer apex) / WRONG_EXTENT / NOT_A_BEND / UNCLEAR) and list every formed bend with no line. Be strict both ways; you won't see the other auditors.`
    const as = (await parallel([1, 2, 3].map(k => () => agent(ap(k), { label: `aud${k}:${c.scan_id}`, phase: 'Audit', schema: AUD, effort: 'high' })))).filter(Boolean)
    if (as.length === 3 && as.every(clean)) return { ...x, as, final: 'GOLD_AS_IS', dir: c.lines_dir }
    const dir = `${OUT}/lines/${c.scan_id}_gold`
    let notes = as.map((a, i) => `auditor ${i + 1}: ${JSON.stringify(a).slice(0, 3500)}`).join('\n')
    for (let round = 1; round <= 2; round++) {
      const fix = await agent(`${CTX}\nGOLD FIXER (max care), round ${round}, scan ${c.scan_id} (part ${c.part_id}, stage '${c.stage}'). Start from ${c.lines_dir}/pairs.json (if any) and prep ${c.prep_dir}. Findings:\n${notes}\nDecide what is really true from the images, write the full corrected pairs.json into ${dir}, run lines into ${dir} under the lock, and check every line with your own close-ups and sections until all land and no formed bend is missed. If impossible, say GIVE_UP with the reason. Return a short report incl. rc.`, { label: `fix${round}:${c.scan_id}`, phase: 'Fix', effort: 'max' })
      if (/GIVE_UP/.test(String(fix))) return { ...x, as, final: 'EXCLUDED', reason: String(fix).slice(0, 800) }
      const vs = (await parallel([1, 2].map(k => () => agent(`${CTX}\nBLIND GOLD VERIFIER ${k} for scan ${c.scan_id}. Lines: ${dir}. Prep ${c.prep_dir}. Do not read fixer notes. Open every render; make your own close-ups and perpendicular sections. pass=true only if EVERY line lands on the outer-skin apex with >= 80% coverage and no formed bend is missed (images_seen > 0).`, { label: `ver${k}r${round}:${c.scan_id}`, phase: 'Fix', schema: VV, effort: 'high' })))).filter(Boolean)
      if (vs.length === 2 && vs.every(v => v.pass && v.images_seen > 0)) return { ...x, as, final: 'GOLD_FIXED', dir }
      notes = `previous fix: ${String(fix).slice(0, 2000)}\nverifiers: ${JSON.stringify(vs).slice(0, 3500)}`
    }
    return { ...x, as, final: 'EXCLUDED', reason: 'failed verification after 2 fix rounds', dir }
  },
  async (x) => {
    // persist each scan's outcome to disk so nothing is lost or truncated
    await agent(`Write this JSON exactly to ${OUT}/outcomes/${x.c.scan_id}.json (mkdir -p first) and reply 'ok'. JSON: ${JSON.stringify({ scan_id: x.c.scan_id, part_id: x.c.part_id, family: x.c.family, stage: x.c.stage, path: x.c.path, has_step: x.c.has_step, cad: x.c.cad_bend_lines ?? -1, prev_truth: x.c.current_truth, final: x.final, dir: x.dir || null, reason: x.reason || null, auditors: (x.as || []).map(a => ({ counts: a.lines.reduce((m, l) => (m[l.verdict] = (m[l.verdict] || 0) + 1, m), {}), missing: a.missing_bends.length, real: a.real_bend_count })) })}`, { label: `save:${x.c.scan_id}`, phase: 'Write', effort: 'low' })
    return { scan_id: x.c.scan_id, final: x.final }
  })

phase('Write')
const fin = await agent(`${CTX}\nTASK: WRITE + MANIFEST. Read every outcome file in ${OUT}/outcomes/*.json (one per scan; do not rely on anything inline). (1) For each GOLD_AS_IS / GOLD_FIXED: write-label --lines <dir> --asset-path <path> --labels-root data/bend_corpus/annotations/ai_bend_lines_gold_v1 (annotator claude-vision-ai-v1; serially under the lock; --confirm-no-bends for zero-bend; for ~/Downloads assets use the identical copy under data/bend_corpus/incoming/2026-10-17_benchmark_v2_downloads/; record refusals). (2) Build ${OUT}/GOLD_SET_V2.json + docs/production/GOLD_SET_V2.md = GOLD_SET_V1 (unchanged, referenced by sha) + these new gold scans, with a clear CAD / NO-CAD split: counts of scans, lines, families, full vs staged for each; the excluded list with reasons; how many prior labels (AI accepted vs legacy AI vs human) needed fixes and what kind. Also write case lists: ${OUT}/case_list_gold_v2_all.json, ${OUT}/case_list_gold_v2_cad.json (production track) and ${OUT}/case_list_gold_v2_nocad.json (blind track). (3) Rescore EXISTING predictions only (${RL}/broad_baseline/runs/best_f1 and runs/runtime via research/broad_baseline/score.py; no engine runs) on each case list; report coverage (scans lacking predictions) and micro + family-macro F1 per lane, plus pre-audit vs gold deltas. Return a concise summary with numbers.`, { label: 'write+manifest', phase: 'Write', effort: 'high' })
return { fin, counts: { candidates: cands.length, as_is: res.filter(r => r && r.final === 'GOLD_AS_IS').length, fixed: res.filter(r => r && r.final === 'GOLD_FIXED').length, excluded: res.filter(r => r && r.final === 'EXCLUDED').length } }
