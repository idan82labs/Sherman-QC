#!/usr/bin/env node
// REAL-part app end-to-end run of the Sherman QC production line (real-wiring audit, 2026-10-19).
//
// Proves the line on a real part with no API shortcut and no synthetic data:
//   - servers start from a TEMPORARY APP COPY (C04 `engine_bundle build --profile app`) with a temp
//     SHERMAN_LINE_DATA_ROOT and inbox; Playwright drives Chromium at 1920x1080;
//   - onboarding of the authorized Gold11 development part 10839000 through the real UI: SKU, its
//     folded STEP, its own drawing PDF (no flat DXF exists for this part), the Vendor Setup studio
//     (confirm the CAD solver's auto-marked bends, order, one מהלך of two bends, save), tolerances, release;
//   - the real FULL scan and the two real stage scans are DROPPED into the hot folder ZEISS-style
//     (temp name, rename, aged mtime); the line service picks them up and runs E3;
//   - the station is opened by the signed-in admin (user token; no device token minted by API);
//   - P1: the station part picker: held-back scans (unknown / unnamed file names) are picked or dismissed at the
//     station in Hebrew (RTL); the assigned job runs E3 and is marked "assigned by operator" (picker.json);
//   - for every scan: station screenshots, DOM values, API result, the PDF downloaded with the
//     station's own button, and a DIRECT E3 CLI run on the job's compiled recipe + the same PLY,
//     then `real_e2e_support.py compare` (engine == job == API == screen == PDF) and `heatmap`
//     (viewer deviations == engine deviations).
// Run the whole driver under the shared geometry lock (it runs engine jobs and the direct E3 runs).
//
// Usage: node scripts/production/run_e2e_real.mjs [--headed] [--keep] [--run NAME] [--scans full,scan_1,scan_2] [--hotfolder]
// Output: docs/production/runlog/e2e/<run>/ (NN-*.png, steps.json, scans/<scan>/{api,dom,direct,compare,pdf}).
// Exit: 0 when every step is PASS, 1 otherwise, 2 setup failure.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { createHash, randomBytes } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { chmod, copyFile, mkdir, mkdtemp, open, readdir, rename, rm, utimes, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import net from 'node:net'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const FE = join(REPO, 'frontend', 'react')
const { probeLabels, labelProblems } = await import(join(FE, 'scripts', 'station-label-probe.mjs'))
const PY = join(REPO, '.venv', 'bin', 'python')
const SUPPORT = join(REPO, 'scripts', 'production', 'run_e2e_app_support.py')
const RSUPPORT = join(REPO, 'scripts', 'production', 'real_e2e_support.py')
const args = process.argv.slice(2)
const HEADED = args.includes('--headed')
const KEEP = args.includes('--keep')
const RUN = args.includes('--run') ? args[args.indexOf('--run') + 1] : `${new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z')}-real`
const HOTFOLDER = args.includes('--hotfolder') // the hotfolder edge-case steps H0-H5 (docs/production/runlog/station_part_picker/HOTFOLDER_E2E.md)
const SCANS_WANTED = (args.includes('--scans') ? args[args.indexOf('--scans') + 1] : 'full,scan_1,scan_2').split(',')
const OUT = join(REPO, 'docs', 'production', 'runlog', 'e2e', RUN)
const STATION = 'cell-1'
const SRC = resolve(REPO, '..', 'source-assets', '02_CAD_QC_3D')
const REAL = {
  stem: '10839000_01', part_number: '10839000', revision: '01', part_name: 'Joggle bracket 10839000',
  step: join(SRC, '10839000_01.stp'),
  // The part's own customer drawing (byte source: the authorized Gold11 dev part's email attachment set).
  drawing: join(REPO, 'data', 'bend_corpus', 'incoming_email_2026-10-17', '20260308_19ccdcb6', 'שרטוט 10839000.pdf'),
  scans: {
    full: { src: join(SRC, '10839000', '10839000_01_full_scan.ply'), zeiss: '10839000_01_FULL_SCAN 1.ply', kind: 'FULL' },
    scan_1: { src: join(SRC, '10839000', '10839000_01_scan_1.ply'), zeiss: '10839000_01_SCAN_1 1.ply', kind: 'PARTIAL' },
    scan_2: { src: join(SRC, '10839000', '10839000_01_scan_2.ply'), zeiss: '10839000_01_SCAN_2 1.ply', kind: 'PARTIAL' },
  },
}
const ADMIN = { username: 'e2e-admin', password: `E2e-${randomBytes(9).toString('base64url')}` }
const sleep = ms => new Promise(r => setTimeout(r, ms))

let chromium
try {
  ;({ chromium } = createRequire(join(FE, 'package.json'))('playwright'))
  if (!existsSync(chromium.executablePath())) throw new Error(`no Chromium at ${chromium.executablePath()}`)
} catch (err) {
  console.error(`run_e2e_real: Playwright Chromium not available (${err.message})`)
  process.exit(2)
}

// ------------------------------------------------------------------ bookkeeping
const steps = []
const failurePages = []
let shotNo = 0
let current = null
function record(status, detail) { current.status = status; if (detail) current.detail = String(detail).slice(0, 3000) }
async function step(id, title, fn, { requires = [] } = {}) {
  current = { id, title, status: 'PASS', assertions: [], screenshots: [], started: new Date().toISOString() }
  steps.push(current)
  const t0 = Date.now()
  const missing = requires.filter(r => !steps.find(s => s.id === r && s.status === 'PASS'))
  if (missing.length) {
    record('NOT_RUN', `prerequisite not met: ${missing.join(', ')}`)
  } else {
    try { await fn(current) } catch (err) {
      record('FAIL', err && err.stack ? err.stack.split('\n').slice(0, 4).join(' | ') : err)
      for (const [label, p] of failurePages) {
        try { await shot(p, `FAIL-${id}-${label}`, 'failure state (diagnostic, not an assertion)') } catch { /* page gone */ }
      }
    }
  }
  current.duration_s = Math.round((Date.now() - t0) / 100) / 10
  console.log(`${current.status.padEnd(10)} ${id} ${title}${current.detail && current.status !== 'PASS' ? `\n           ${current.detail.slice(0, 400)}` : ''}`)
  await writeFile(join(OUT, 'steps.json'), JSON.stringify({ run: RUN, steps }, null, 2))
}
function check(cond, what, value) {
  current.assertions.push({ what, ok: !!cond, value: value === undefined ? null : value })
  assert.ok(cond, `${what}${value === undefined ? '' : ` (${JSON.stringify(value).slice(0, 400)})`}`)
}
function note(cond, what, value) {
  current.assertions.push({ what, ok: !!cond, value: value === undefined ? null : value, informational: true })
}
async function settle(p, timeoutMs = 5000) {
  await p.waitForFunction(() => document.getAnimations().every(a => a.playState !== 'running'
    || (a.effect && a.effect.getComputedTiming && a.effect.getComputedTiming().iterations === Infinity)), null, { timeout: timeoutMs }).catch(() => {})
  await p.waitForFunction(() => !document.querySelector('[data-wizard][data-loaded="false"]'), null, { timeout: timeoutMs }).catch(() => {})
}
async function shot(page, name, what) {
  shotNo += 1
  const file = `${String(shotNo).padStart(2, '0')}-${name}.png`
  await page.screenshot({ path: join(OUT, file) })
  current.screenshots.push({ file, pairs_with: what })
  return file
}

// ------------------------------------------------------------------ processes
const children = []
function run(cmd, argv, opts = {}) {
  const r = spawnSync(cmd, argv, { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024, ...opts })
  return { code: r.status, out: r.stdout || '', err: r.stderr || '' }
}
const lastJson = out => { try { return JSON.parse(out.trim().split('\n').pop() || '{}') } catch { return {} } }
function start(name, argv, env, cwd) {
  const child = spawn(PY, argv, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'], detached: false })
  const chunks = []
  child.stdout.on('data', d => chunks.push(d))
  child.stderr.on('data', d => chunks.push(d))
  child.logText = () => Buffer.concat(chunks).toString('utf8')
  child.name = name
  children.push(child)
  return child
}
async function freePort() {
  return new Promise((res, rej) => {
    const s = net.createServer()
    s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)) })
    s.on('error', rej)
  })
}
async function waitFor(fn, timeoutMs, what, intervalMs = 200) {
  const t0 = Date.now()
  let last
  while (Date.now() - t0 < timeoutMs) {
    try { last = await fn(); if (last) return last } catch (e) { last = e }
    await sleep(intervalMs)
  }
  throw new Error(`timeout waiting for ${what} (${last instanceof Error ? last.message : JSON.stringify(last)})`)
}
const sha256 = f => createHash('sha256').update(readFileSync(f)).digest('hex')

// ------------------------------------------------------------------ setup
await mkdir(join(OUT, 'logs'), { recursive: true })
const WORK = await mkdtemp(join(process.env.TMPDIR || tmpdir(), 'sherman-e2e-real-'))
const APP = join(WORK, 'app')
const DATA = join(WORK, 'line')
const INBOX = join(WORK, 'inbox')
const CUSTODY = join(WORK, 'real')
await mkdir(INBOX, { recursive: true })
console.log(`run ${RUN}\n  out  ${OUT}\n  work ${WORK}`)

let base = ''
let token = ''
let partId = ''
let realSku = ''
let recipeId = ''
let nBends = 0
let studioLabelBySeq = {} // sequence_index -> the studio (CAD) label the operator tagged
let browser = null
const api = async (method, path, body, tok = token) => {
  const r = await fetch(`${base}${path}`, {
    method, headers: { ...(tok ? { Authorization: `Bearer ${tok}` } : {}), ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  })
  const text = await r.text()
  let json = null
  try { json = JSON.parse(text) } catch { /* not json */ }
  return { status: r.status, json, text }
}
// ZEISS-like drop (what the scanner PC does): write to a hidden temp name in the inbox, rename, age the mtime.
async function dropFile(src, fileName) {
  const tmp = join(INBOX, `.${fileName}.part`)
  await copyFile(src, tmp)
  await chmod(tmp, 0o644)
  await rename(tmp, join(INBOX, fileName))
  const old = new Date(Date.now() - 5000)
  await utimes(join(INBOX, fileName), old, old)
  return sha256(src)
}
// Observation only (read-only GETs): which job the drop produced and when it finished.
const jobSha = new Map()
async function jobForSha(sha, timeoutMs = 120000) {
  return waitFor(async () => {
    const r = await api('GET', `/api/line/jobs?station_id=${STATION}&limit=50`)
    for (const j of (r.json && r.json.jobs) || []) {
      if (!jobSha.has(j.job_id)) {
        const d = await api('GET', `/api/line/jobs/${j.job_id}`)
        if (d.json && d.json.job) jobSha.set(j.job_id, d.json.job.scan_sha256)
      }
      if (jobSha.get(j.job_id) === sha) return j
    }
    return null
  }, timeoutMs, `a job for scan ${sha.slice(0, 12)}`)
}
async function jobDone(jobId, timeoutMs = 900000) {
  return waitFor(async () => {
    const r = await api('GET', `/api/line/jobs/${jobId}`)
    const job = r.json && (r.json.job || r.json)
    return job && ['completed', 'failed', 'cancelled'].includes(job.execution_status) ? job : null
  }, timeoutMs, `job ${jobId} to finish`, 500)
}
const baseEnv = {
  PATH: process.env.PATH, HOME: process.env.HOME, TMPDIR: process.env.TMPDIR || tmpdir(), LANG: 'en_US.UTF-8',
  PYTHONPATH: `${APP}:${join(APP, 'backend')}`, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', OMP_NUM_THREADS: '1',
  SHERMAN_LINE_ENABLED: '1', SHERMAN_LINE_DATA_ROOT: DATA, SHERMAN_LINE_STATION_ID: STATION, SHERMAN_LINE_INBOX: INBOX,
  SECRET_KEY: randomBytes(32).toString('hex'), JWT_EXPIRATION_HOURS: '4',
}
const repoEnv = { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend`, PYTHONDONTWRITEBYTECODE: '1', OMP_NUM_THREADS: '1' }

await step('S0', 'custody copies of the real part (read-only, byte-identical) and a temporary app copy (C04 --profile app)', async () => {
  await mkdir(CUSTODY, { recursive: true })
  const custody = {}
  const copyRo = async (src, name) => {
    const dst = join(CUSTODY, name)
    await copyFile(src, dst); await chmod(dst, 0o444)
    custody[name] = { source: src, sha256: sha256(dst), identical: sha256(src) === sha256(dst) }
    return dst
  }
  check(existsSync(REAL.step) && existsSync(REAL.drawing), 'authorized 10839000 STEP and drawing PDF present')
  await copyRo(REAL.step, `${REAL.stem}.stp`)
  await copyRo(REAL.drawing, `${REAL.part_number}_drawing.pdf`)
  for (const k of SCANS_WANTED) {
    check(existsSync(REAL.scans[k].src), `scan ${k} present`, REAL.scans[k].src)
    await copyRo(REAL.scans[k].src, `${k}.ply`)
  }
  current.custody = custody
  check(Object.values(custody).every(c => c.identical), 'read-only temp copies are byte-identical to the sources', custody)
  const b = run(PY, ['-m', 'apps.production_line.engine_bundle', 'build', '--repo-root', REPO, '--dest', APP, '--profile', 'app'], { cwd: REPO, env: repoEnv, timeout: 900000 })
  await writeFile(join(OUT, 'logs', 'build-app-copy.log'), b.out + b.err)
  check(b.code === 0, 'app copy built (exit 0)', b.code)
  const manifest = JSON.parse(readFileSync(join(APP, 'bundle_manifest.json'), 'utf8'))
  const mfiles = Array.isArray(manifest.files) ? manifest.files.map(f => f.path || f) : Object.keys(manifest.files || {})
  current.bundle = { bundle_id: manifest.bundle_id, profile: manifest.profile, files: mfiles.length }
  const hasLegacy = mfiles.some(f => /bend_analysis_worker/.test(f))
  current.bundle.has_legacy_worker = hasLegacy
  check(existsSync(join(APP, 'frontend', 'dist', 'index.html')), 'copy has a built SPA')
})

let server = null
let service = null
await step('S1', 'bootstrap admin, start the API server and the line service from the app copy', async () => {
  const boot = spawnSync(PY, ['-m', 'apps.production_line.security.bootstrap_admin', '--users-db', join(APP, 'data', 'users.db'),
    '--username', ADMIN.username, '--password-stdin'], { cwd: APP, env: baseEnv, input: `${ADMIN.password}\n`, encoding: 'utf8' })
  check(boot.status === 0, 'bootstrap_admin exit 0 (first-admin provisioning is a CLI step by design)', boot.stderr.slice(-300))
  const port = await freePort()
  base = `http://127.0.0.1:${port}`
  server = start('server', ['-m', 'apps.production_line.server', '--port', String(port)], baseEnv, APP)
  service = start('service', ['-m', 'apps.production_line.service_main', '--build-bundle'], baseEnv, APP)
  const ready = await waitFor(async () => {
    const r = await api('GET', '/api/line/health/ready', null, '')
    return r.json && r.json.ready === true ? r.json : null
  }, 300000, 'GET /api/line/health/ready -> ready')
  check(ready.ready === true, 'line ready', ready)
}, { requires: ['S0'] })

browser = await chromium.launch({ headless: !HEADED, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1, hasTouch: true, acceptDownloads: true, locale: 'en-US' })
const page = await context.newPage()
page.on('dialog', d => d.accept().catch(() => {}))
failurePages.push(['wizard', page])
const pageErrors = []
page.on('pageerror', e => pageErrors.push(String(e)))
page.setDefaultTimeout(30000)
const wizardFor = (id, s) => `${base}/production/parts/${encodeURIComponent(id)}/onboard?step=${s}`
const attr = async (p, sel, name) => p.locator(sel).first().getAttribute(name)
const fileInput = p => p.locator('[data-step-body] input[type="file"], input[type="file"]').first()
const stepOf = p => new URL(p.url()).searchParams.get('step')

await step('W1', 'admin logs in; Production -> Parts -> New part 10839000; upload its folded STEP (CAD solver auto-marks bends)', async () => {
  await page.goto(`${base}/login`)
  await page.fill('#username', ADMIN.username)
  await page.fill('#password', ADMIN.password)
  await page.click('button[type="submit"]')
  await page.waitForURL(u => !u.pathname.startsWith('/login'), { timeout: 20000 })
  token = await page.evaluate(() => localStorage.getItem('token'))
  check(!!token, 'login stored a user token')
  await page.click('aside nav a[href="/production/parts"]')
  await page.locator('[data-production-parts]').waitFor()
  await page.locator('[data-language="en"]').first().click().catch(() => {})
  await page.click('[data-new-part]')
  await page.locator('[data-new-production-part]').waitFor()
  await page.fill('[data-field="part_number"]', REAL.part_number)
  await page.fill('[data-field="part_name"]', REAL.part_name)
  await page.fill('[data-field="revision"]', REAL.revision)
  realSku = await attr(page, '[data-sku-preview]', 'data-sku-preview')
  check(realSku === REAL.part_number, 'SKU preview = 10839000', realSku)
  await shot(page, 'new-part-form', 'SKU preview attribute = 10839000')
  await page.click('[data-create-part]')
  await page.waitForURL(/\/production\/parts\/[^/]+\/onboard\?step=cad$/)
  partId = decodeURIComponent(new URL(page.url()).pathname.split('/')[3])
  await fileInput(page).setInputFiles(join(CUSTODY, `${REAL.stem}.stp`))
  await page.locator('[data-cad-current]').waitFor({ timeout: 600000 })
  const n = await waitFor(async () => {
    const t = (await page.locator('[data-candidate-count]').first().innerText()).trim()
    return /^\d+$/.test(t) && Number(t) > 0 ? Number(t) : null
  }, 300000, 'the CAD candidate count')
  await page.locator('[data-upload-phase]').waitFor({ state: 'detached', timeout: 60000 }).catch(() => {})
  const vs = (await api('GET', `/api/parts/${encodeURIComponent(partId)}/vendor-setup`)).json
  current.cad = { candidates: n, candidate_generation: vs && vs.candidate_generation, labels: ((vs && vs.candidates) || []).map(c => `${c.label}:${c.target_angle}`) }
  check(n > 0 && vs && vs.candidates.length === n, 'the CAD bend solver auto-marked bends on the real STEP (wizard count = API candidates)', current.cad)
  const cadName = (await page.locator('[data-cad-current]').innerText()).replace(/\s+/g, ' ').trim()
  current.cad_card = cadName
  check(cadName.includes(REAL.stem), 'the CAD card names the uploaded file (not the stored hash name)', cadName)
  nBends = n
  await settle(page)
  await shot(page, 'wizard-cad', `candidate count = ${n} (= API candidates); card names ${cadName}`)
}, { requires: ['S1'] })

await step('W2', 'upload the part\'s drawing PDF; "Next" goes to the optional flat step (not skipped)', async () => {
  await page.goto(wizardFor(partId, 'drawing'))
  await fileInput(page).setInputFiles(join(CUSTODY, `${REAL.part_number}_drawing.pdf`))
  await page.locator('[data-drawing-current]').waitFor({ timeout: 60000 })
  await page.locator('[data-upload-phase]').waitFor({ state: 'detached', timeout: 60000 }).catch(() => {})
  const name = (await page.locator('[data-drawing-current]').innerText()).replace(/\s+/g, ' ').trim()
  current.drawing_card = name
  check(name.includes(REAL.part_number) && !/\b[0-9a-f]{24,}\.pdf/i.test(name), 'the drawing card names the uploaded file (not the stored hash name)', name)
  await settle(page)
  await shot(page, 'wizard-drawing', `drawing card: ${name}`)
  // The rail's "Next" pill must point at the optional unfolded-part step too (optional is not skipped).
  current.rail_next = await attr(page, '.onb-rail [data-next="true"]', 'data-step')
  check(current.rail_next === 'flat', 'the rail suggests the optional unfolded-part step next', current.rail_next)
  await page.click('footer [data-next]')
  await page.waitForURL(u => new URL(u).searchParams.get('step') !== 'drawing', { timeout: 15000 })
  check(stepOf(page) === 'flat', '"Next" from the drawing step opens the optional unfolded-part (flat) step', stepOf(page))
  await page.locator('[data-step-body="flat"]').waitFor({ timeout: 30000 })
  await settle(page)
  await shot(page, 'wizard-flat-optional', 'flat step shown (optional); 10839000 has no flat DXF on file')
  await page.click('footer [data-next]')
  await page.waitForURL(u => new URL(u).searchParams.get('step') === 'bends', { timeout: 15000 })
  check(stepOf(page) === 'bends', 'skipping the optional flat step goes to Bends & order', stepOf(page))
}, { requires: ['W1'] })

// ------------------------------------------------------------------ Vendor Setup studio (as run_e2e_app.mjs driveStudio)
const STUDIO = '[data-studio-body]'
async function clickBend(p, id, label, verb, clicks) {
  const marker = p.locator(`${STUDIO} [data-bend-marker][aria-label="${verb} ${label} on model"]`).first()
  try {
    await marker.click({ timeout: 4000 })
    clicks.push({ id, label, verb, via: 'model' })
  } catch {
    await p.locator(`${STUDIO} [data-bend-id="${id}"] .vendor-setup-select-bend`).click()
    clicks.push({ id, label, verb, via: 'list' })
  }
}
async function studioRows(p) {
  return p.locator(`${STUDIO} [data-bend-id]`).evaluateAll(els => els.map(e => ({
    id: e.getAttribute('data-bend-id'), label: e.getAttribute('data-bend-label'), review: e.getAttribute('data-review-status'),
    status: e.getAttribute('data-mark-status'), step: e.getAttribute('data-vendor-step') })))
}

await step('W3', 'Vendor Setup studio: confirm every auto-marked bend ON THE MODEL, tag the order, one מהלך of two bends, save, back to the wizard', async () => {
  const p = page
  const did = { clicks: [] }
  current.ui = did
  await p.goto(wizardFor(partId, 'bends'))
  const entry = p.locator('[data-step-body] [data-tag-bends], [data-tag-bends]').first()
  await entry.waitFor({ timeout: 60000 })
  await shot(p, 'bends-entry', 'wizard step 5 with the "Tag bends" action')
  await entry.click()
  await p.waitForURL(u => new URL(u).searchParams.get('studio') === '1', { timeout: 15000 })
  await p.locator(`${STUDIO} [data-studio-bar]`).waitFor({ timeout: 60000 })
  await p.locator(`${STUDIO} [data-bend-id]`).first().waitFor({ timeout: 120000 })
  const setup0 = (await api('GET', `/api/parts/${encodeURIComponent(partId)}/vendor-setup`)).json
  const cands = (setup0 && setup0.candidates) || []
  const rows0 = (await studioRows(p)).filter(r => r.status === 'active')
  did.candidates = cands.map(c => ({ feature_id: c.feature_id, label: c.label, target_angle: c.target_angle }))
  check(rows0.length > 0 && rows0.length === cands.length && rows0.every(r => cands.some(c => c.feature_id === r.id)),
    'studio lists every bend the CAD solver auto-marked (DOM rows = API candidates)', { dom: rows0.length, api: cands.length })
  check(rows0.every(r => r.review === 'pending'), 'auto-marked bends start unconfirmed (pending)', rows0.map(r => r.review))
  const markers = await waitFor(async () => {
    const n = await p.locator(`${STUDIO} [data-bend-marker]`).count()
    return n === cands.length ? n : null
  }, 60000, 'one on-model marker per auto-marked bend').catch(() => p.locator(`${STUDIO} [data-bend-marker]`).count())
  check(markers === cands.length, 'the CAD view draws one numbered marker per auto-marked bend', { markers, candidates: cands.length })
  await sleep(800)
  await shot(p, 'studio-auto-marked', `${cands.length} auto-marked bends, all pending, one marker each`)
  const labelOf = new Map(rows0.map(r => [r.id, r.label]))
  // The shop's real bending order for 10839000, as the operator knows it and as the physical stage
  // scans show it (scan_1: the four joggle bends + both short walls with their lips = 8 formed;
  // scan_2: + one long wall and its lip = 10; full: + the other long wall and lip). The joggle pair
  // B1+B2 is one press stroke (מהלך). Labels are the CAD solver's candidate labels.
  const REAL_ORDER = ['B1', 'B2', 'B3', 'B4', 'B5', 'B10', 'B7', 'B12', 'B6', 'B11', 'B8', 'B9']
  const idOf = new Map(rows0.map(r => [r.label, r.id]))
  check(REAL_ORDER.every(l => idOf.has(l)) && REAL_ORDER.length === rows0.length, 'the CAD solver labelled the 12 bends B1..B12 as expected', rows0.map(r => r.label))
  const order = REAL_ORDER.map(l => idOf.get(l))
  const moves = [order.slice(0, 2)]
  did.order = order.map(id => labelOf.get(id))
  did.moves = moves.map(m => m.map(id => labelOf.get(id)))
  for (const id of order) {
    await clickBend(p, id, labelOf.get(id), 'Select bend', did.clicks)
    await p.locator(`${STUDIO} [data-confirm-bend="${id}"]`).click({ timeout: 15000 })
    await waitFor(async () => (await attr(p, `${STUDIO} [data-bend-id="${id}"]`, 'data-review-status')) === 'confirmed', 10000, `bend ${labelOf.get(id)} confirmed`)
  }
  const rows1 = await studioRows(p)
  check(rows1.filter(r => r.status === 'active').every(r => r.review === 'confirmed'), 'every auto-marked bend shows confirmed')
  await shot(p, 'studio-confirmed', 'all rows confirmed')
  await p.locator(`${STUDIO} [data-tag-order]`).click()
  await p.locator(`${STUDIO} [data-order-mode="true"]`).waitFor({ timeout: 10000 })
  let moveNo = 0
  for (const [i, id] of order.entries()) {
    await clickBend(p, id, labelOf.get(id), 'Tag bend', did.clicks)
    await p.locator(`${STUDIO} [data-order-assigned="${id}"]`).waitFor({ timeout: 10000 })
    const mv = moves.find(m => m[m.length - 1] === id)
    if (mv) {
      await p.locator(`${STUDIO} [data-tag-move]`).click()
      moveNo += 1
      await p.locator(`${STUDIO} [data-machine-move="${moveNo}"]`).waitFor({ timeout: 10000 })
    } else if (!moves.some(m => m.includes(id)) && i < order.length - 1) {
      await p.locator(`${STUDIO} [data-leave-without-move]`).click()
    }
  }
  const move1 = (await p.locator(`${STUDIO} [data-machine-move="1"]`).first().innerText()).trim()
  check(moves[0].every(id => move1.includes(labelOf.get(id))), 'מהלך 1 names its two bends', move1)
  await shot(p, 'studio-order-moves', `order tagged; ${move1.replace(/\s+/g, ' ')}`)
  await p.locator(`${STUDIO} [data-apply-order]`).click()
  await p.locator(`${STUDIO} [data-order-mode="false"][data-order-confirmed="true"]`).waitFor({ timeout: 10000 })
  did.via = { model: did.clicks.filter(c => c.via === 'model').length, list: did.clicks.filter(c => c.via === 'list').length }
  check(did.via.list === 0, 'every bend was clicked on the CAD model (no list fallback)', did.via)
  await p.locator(`${STUDIO} [data-save-draft]`).click()
  await waitFor(async () => (await attr(p, `${STUDIO} .onb-save-state`, 'data-dirty')) === 'false', 30000, 'the studio to show "all saved"')
  check(await p.locator(`${STUDIO} [role="alert"]`).count() === 0, 'save raised no error')
  await shot(p, 'studio-saved', 'saved, no error')
  await p.locator(`${STUDIO} [data-studio-done]`).click()
  await p.waitForURL(u => new URL(u).searchParams.get('studio') !== '1', { timeout: 15000 })
  await p.locator('[data-bend-summary]').waitFor({ timeout: 60000 })
  const d = (await api('GET', `/api/parts/${encodeURIComponent(partId)}/vendor-setup`)).json.draft
  did.draft = d && { sequence: d.sequence.map(id => labelOf.get(id) || id), machine_moves: (d.machine_moves || []).map(m => m.map(id => labelOf.get(id) || id)), operator_sequence_confirmed: d.operator_sequence_confirmed }
  check(d && !d.is_stale && JSON.stringify(d.sequence) === JSON.stringify(order), 'saved draft (read back): order = the clicked order', did.draft)
  check(d && JSON.stringify((d.machine_moves || []).filter(m => m.length > 1)) === JSON.stringify(moves), 'saved draft: מהלך group = the tagged pair', did.draft)
  check(d && d.marks.filter(m => m.status === 'active').every(m => m.review_status === 'confirmed' && m.source === 'cad_candidate'), 'saved draft: every CAD mark confirmed, no manual marks')
  await settle(page)
  await shot(p, 'wizard-bends-order', 'wizard step 5 after Done')
}, { requires: ['W2'] })

// General measurement (GENERAL_MEASUREMENT, kind bend_pair) authored in the studio by clicking:
// "Measure bends", two bends on the model, then two points on the CAD surface next to each bend.
let gmLabel = ''
await step('W3b', 'Vendor Setup studio: author one general measurement between two parallel bends (bend_pair) by clicking on the CAD, save', async () => {
  const p = page
  await p.goto(wizardFor(partId, 'bends'))
  await p.locator('[data-step-body] [data-tag-bends], [data-tag-bends]').first().click()
  await p.locator(`${STUDIO} [data-studio-bar]`).waitFor({ timeout: 60000 })
  await p.locator(`${STUDIO} [data-bend-id]`).first().waitFor({ timeout: 120000 })
  await waitFor(async () => (await p.locator(`${STUDIO} [data-bend-marker]`).count()) === nBends, 60000, 'markers')
  await sleep(1000)
  const setup = (await api('GET', `/api/parts/${encodeURIComponent(partId)}/vendor-setup`)).json
  const cands = setup.candidates
  // Two parallel bends with the largest separation (a clearly two-sided pair; the recipe nominal is the CAD distance).
  let best = null
  for (const a of cands) for (const b of cands) {
    if (a.feature_id >= b.feature_id || !a.line_start || !b.line_start) continue
    const da = a.line_end.map((v, i) => v - a.line_start[i]); const db = b.line_end.map((v, i) => v - b.line_start[i])
    const na = Math.hypot(...da); const nb = Math.hypot(...db)
    const cos = Math.abs(da.reduce((s2, v, i) => s2 + v * db[i], 0) / (na * nb))
    if (cos < 0.999 || Math.abs(Number(a.target_angle) - Number(b.target_angle)) > 1e-6) continue
    const ma = a.line_start.map((v, i) => (v + a.line_end[i]) / 2); const mb = b.line_start.map((v, i) => (v + b.line_end[i]) / 2)
    const dist = Math.hypot(...ma.map((v, i) => v - mb[i]))
    if (!best || dist > best.dist) best = { a, b, dist }
  }
  check(!!best, 'found a pair of parallel bends to measure between', best && [best.a.label, best.b.label])
  current.pair = [best.a.label, best.b.label]
  await p.getByRole('button', { name: /^Between two bends$/ }).click()
  await p.locator(`${STUDIO} [data-measurement-mode="bend_pair"]`).waitFor({ timeout: 10000 })
  const clicks = []
  for (const c of [best.a, best.b]) await clickBend(p, c.feature_id, c.label, 'Select bend', clicks)
  current.bend_clicks = clicks
  check(clicks.every(c => c.via === 'model'), 'both bends picked on the model', clicks)
  // Points: click the CAD surface a few pixels beside each bend's on-model marker (the marker itself is a button).
  const canvas = p.locator(`${STUDIO} .vendor-setup-canvas canvas`).first()
  const status = () => p.locator(`${STUDIO} [data-measurement-mode="bend_pair"] [role="status"]`).innerText()
  const picked = []
  // The marker's leader line starts ON the bend line ("M anchorX anchorY L ..."): pick the CAD surface just beside that anchor.
  const anchorOf = label => p.evaluate(l => {
    const all = [...document.querySelectorAll('[data-studio-body] [data-bend-marker]')]
    const btn = all.find(e => e.getAttribute('aria-label') === `Select bend ${l} on model`)
      || all.find(e => new RegExp(`(^|\\W)${l}(\\W|$)`).test(e.getAttribute('aria-label') || '') || (e.textContent || '').trim().endsWith(l))
    if (!btn) return { missing: true, labels: all.map(e => e.getAttribute('aria-label')) }
    const host = btn.parentElement.parentElement.parentElement
    const idx = [...btn.parentElement.children].indexOf(btn)
    const path = host.querySelector('svg').children[idx].children[1]
    const r = host.getBoundingClientRect()
    const m = /^M\s+([-\d.]+)\s+([-\d.]+)/.exec(path.getAttribute('d') || '')
    if (!m || path.style.display === 'none') { const b = btn.getBoundingClientRect(); return { x: b.x + b.width / 2, y: b.y + b.height / 2, leader: false } }
    return { x: r.x + Number(m[1]), y: r.y + Number(m[2]), leader: true }
  }, label)
  // Both anchors are read before the first point: once a point is picked, the markers turn into plain labels.
  const anchors = new Map()
  for (const c of [best.a, best.b]) anchors.set(c.label, await anchorOf(c.label))
  current.anchors = Object.fromEntries(anchors)
  check([...anchors.values()].every(a2 => !a2.missing), 'both bend markers (and their on-line anchors) found', current.anchors)
  for (const c of [best.a, best.b]) {
    const anc = anchors.get(c.label)
    const before = await status()
    let ok = false
    const ring = []
    for (const rad of [10, 16, 24]) for (let k = 0; k < 8; k++) ring.push([Math.round(rad * Math.cos(k * Math.PI / 4)), Math.round(rad * Math.sin(k * Math.PI / 4))])
    for (const [dx, dy] of ring) {
      const x = anc.x + dx; const y = anc.y + dy
      await p.mouse.click(x, y)
      await sleep(400)
      if ((await status()) !== before) { ok = true; picked.push({ bend: c.label, anchor: anc, dx, dy }); break }
    }
    check(ok, `a CAD surface point was picked next to ${c.label}`, picked)
  }
  current.points = picked
  const distText = await p.locator(`${STUDIO} [data-measurement-mode="bend_pair"]`).innerText()
  current.cad_distance_text = distText.replace(/\s+/g, ' ').slice(0, 300)
  gmLabel = `E2E ${best.a.label}-${best.b.label}`
  await p.fill('#reference-label', gmLabel)
  await shot(p, 'studio-measurement-picked', `bend_pair ${best.a.label} ↔ ${best.b.label}: two points picked on the CAD`)
  await p.getByRole('button', { name: /^Add reference$/ }).click()
  await p.locator(`${STUDIO} [data-general-measurement]`).first().waitFor({ timeout: 10000 })
  await p.locator(`${STUDIO} [data-save-draft]`).click()
  await waitFor(async () => (await attr(p, `${STUDIO} .onb-save-state`, 'data-dirty')) === 'false', 30000, 'saved')
  await shot(p, 'studio-measurement-saved', 'general measurement saved in the draft')
  await p.locator(`${STUDIO} [data-studio-done]`).click()
  await p.waitForURL(u => new URL(u).searchParams.get('studio') !== '1', { timeout: 15000 })
  const d = (await api('GET', `/api/parts/${encodeURIComponent(partId)}/vendor-setup`)).json.draft
  current.saved = (d.general_measurements || []).map(g => ({ label: g.label, kind: g.kind, bend_ids: g.bend_ids, cad_distance: g.cad_distance }))
  check(current.saved.length === 1 && current.saved[0].kind === 'bend_pair' && current.saved[0].label === gmLabel, 'draft holds the bend_pair measurement', current.saved)
  const g = d.general_measurements[0]
  const distToLine = (pt, c) => { const a0 = c.line_start; const u = c.line_end.map((v, i) => v - a0[i]); const n0 = Math.hypot(...u); const r = pt.map((v, i) => v - a0[i]); const t = r.reduce((s2, v, i) => s2 + v * u[i], 0) / n0; return Math.sqrt(Math.max(0, r.reduce((s2, v) => s2 + v * v, 0) - t * t)) }
  current.point_owner = [g.point_start, g.point_end].map(pt => [best.a, best.b].map(c => `${c.label}:${distToLine(pt, c).toFixed(1)}mm`))
  const own = [g.point_start, g.point_end].map(pt => distToLine(pt, best.a) < distToLine(pt, best.b) ? best.a.label : best.b.label)
  check(own[0] !== own[1], 'the two picked points sit next to different bends (one per bend)', current.point_owner)
  check(!d.is_stale && d.operator_sequence_confirmed, 'draft still fresh with the order confirmed')
}, { requires: ['W3'] })

await step('W4', 'tolerances (FINAL), review, release 10839000@v1 (E1 compile from the saved setup), expectations table', async () => {
  const p = page
  await p.goto(wizardFor(partId, 'tolerances'))
  await p.locator('[data-tolerance-editor]').waitFor()
  await waitFor(async () => (await p.locator('[data-override-row]').count()) === nBends, 60000, `the ${nBends} per-bend rows`)
  await sleep(500)
  for (const [id, v] of [['angle', '1'], ['flange', '0.5'], ['bendLine', '0.5'], ['backgauge', '0.3']]) {
    await p.fill(`#${id}-plus`, v)
    if (await p.locator(`#${id}-minus`).count()) await p.fill(`#${id}-minus`, v)
  }
  await p.selectOption('[data-tol-source]', 'SHOP_STANDARD')
  await p.check('[data-stage="FINAL"]')
  await p.click('[data-save-tolerances]')
  await p.locator('[data-tolerances-saved]').waitFor({ timeout: 30000 })
  await shot(p, 'tolerances', 'tolerances saved')
  await p.goto(wizardFor(partId, 'review'))
  await p.locator('[data-wizard][data-loaded="true"]').waitFor({ timeout: 60000 })
  await p.locator('[data-review-step]').waitFor()
  await settle(p)
  await shot(p, 'review', 'review step')
  await p.goto(wizardFor(partId, 'release'))
  await p.locator('[data-release-button]').waitFor()
  await p.click('[data-release-button]')
  await p.locator('[data-release-result="released"], [data-release-refused]').first().waitFor({ timeout: 300000 })
  if (await p.locator('[data-release-refused]').count()) {
    current.release_refusal = (await p.locator('[data-release-refused]').innerText()).slice(0, 1500)
    await shot(p, 'release-refused', 'release refused')
    check(false, 'release succeeded', current.release_refusal)
  }
  recipeId = `${realSku}@v1`
  check((await p.locator('[data-release-result]').innerText()).includes(recipeId), `release names ${recipeId}`)
  await p.locator('[data-expectations-status="READY"]').first().waitFor({ timeout: 120000 })
  check(await p.locator('[data-expectation-row]').count() === nBends, `expectations table has ${nBends} rows`)
  await settle(p)
  await shot(p, 'released', `released ${recipeId}`)
  await p.locator('[data-expectation-row]').first().scrollIntoViewIfNeeded()
  await shot(p, 'released-expectations', 'nominal flange lengths and angles')
  current.expectations = await p.locator('[data-expectation-row]').evaluateAll(els => els.map(e => e.innerText.replace(/\s+/g, ' ').trim()))
  const rec = (await api('GET', `/api/production/recipes/${encodeURIComponent(recipeId)}`)).json
  current.recipe = rec && { compiled_part_sha256: rec.compiled_part_sha256 || (rec.recipe && rec.recipe.compiled_part_sha256), bends: rec.recipe && rec.recipe.bends && rec.recipe.bends.length, benchmark: rec.benchmark_reference }
  const sv = (await api('GET', `/api/parts/${encodeURIComponent(partId)}/vendor-setup`)).json
  const seqLabels = (sv.draft.sequence || []).map(id => (sv.candidates.find(c => c.feature_id === id) || {}).label)
  const recBends = ((rec && rec.recipe && rec.recipe.bends) || []).slice().sort((a, b) => a.sequence_index - b.sequence_index)
  current.recipe_order = recBends.map(b => `${b.sequence_index}:${b.label}:${b.machine_move_index ?? b.machine_move ?? ''}`)
  // The recipe renumbers bends by sequence (label B<sequence_index>); cad_label keeps the studio's CAD label.
  current.recipe_order = recBends.map(b => `${b.sequence_index}:${b.label}<-${b.cad_label}:${b.machine_move_index ?? ''}`)
  check(recBends.length === nBends && recBends.every((b, i) => b.cad_label === seqLabels[i]), 'the released recipe carries the studio order (sequence_index = clicked order, by CAD label)', { recipe: current.recipe_order, studio: seqLabels })
  // The recipe renumbers bends by sequence; the station and the PDF show cad_label (via result.context.bend_display_labels).
  studioLabelBySeq = Object.fromEntries(recBends.map(b => [b.sequence_index, b.cad_label]))
}, { requires: ['W3'] })

// ------------------------------------------------------------------ station (same signed-in admin; user token)
const station = await context.newPage()
failurePages.push(['station', station])
station.on('pageerror', e => pageErrors.push(`station: ${String(e)}`))
station.setDefaultTimeout(30000)
const sattr = (sel, name) => attr(station, sel, name)

await step('K1', 'open the station screen as the signed-in admin (no device token minted through the API)', async () => {
  await station.goto(`${base}/station/${STATION}`)
  await station.locator('[data-testid="station-root"]').waitFor({ timeout: 30000 })
  check(await station.locator('[data-testid="station-auth-gate"]').count() === 0, 'no token gate: the station accepts the signed-in user')
  const conn = await waitFor(async () => (await sattr('[data-testid="station-root"]', 'data-connection')) === 'live' ? 'live' : null, 20000, 'live').catch(async () => sattr('[data-testid="station-root"]', 'data-connection'))
  note(conn === 'live', 'station on the live stream', conn)
  await settle(station)
  await shot(station, 'station-empty', `station connected (${conn})`)
}, { requires: ['S1'] })

async function canvasPixels(p) {
  return p.evaluate(() => {
    const c = document.querySelector('[data-testid="station-viewer"] canvas')
    if (!c) return { painted: 0, colours: 0 }
    const probe = document.createElement('canvas')
    probe.width = 240; probe.height = 160
    const ctx = probe.getContext('2d')
    ctx.drawImage(c, 0, 0, probe.width, probe.height)
    const d = ctx.getImageData(0, 0, probe.width, probe.height).data
    let painted = 0
    const colours = new Set()
    for (let i = 0; i < d.length; i += 4) { if (d[i + 3] > 0) painted++; colours.add(`${d[i] >> 4},${d[i + 1] >> 4},${d[i + 2] >> 4}`) }
    return { painted, colours: colours.size }
  })
}
async function stationDom(p) {
  return p.evaluate(() => {
    const txt = el => (el ? el.textContent.trim() : null)
    // Station v2 bend list: one [data-testid=bend-row] per bend; the flange / angle cells keep their test ids.
    const rows = [...document.querySelectorAll('[data-testid="bend-row"]')].map(row => {
      const seq = row.getAttribute('data-sequence')
      const cells = {}
      for (const k of ['flange-a', 'flange-b', 'angle']) {
        const td = row.querySelector(`[data-testid="cell-${seq}-${k}"]`)
        if (td) cells[k] = { state: td.getAttribute('data-state'), decision: td.getAttribute('data-decision'), actual: txt(td.querySelector('.st-actual')), delta: txt(td.querySelector('.st-delta')) }
      }
      return { seq: Number(seq), uid: row.getAttribute('data-bend-uid'), label: txt(row.querySelector('.st-bend-label')), verdict: row.getAttribute('data-verdict'),
        unformed: row.classList.contains('is-unformed') && !!row.querySelector(`[data-testid="cell-${seq}-unformed"]`), cells }
    })
    const gms = [...document.querySelectorAll('.st-gm-row')].map(row => {
      const td = row.querySelector('[data-testid^="gm-"]')
      return { label: txt(row.querySelector('.st-gm-label')), id: td && td.getAttribute('data-testid').slice(3), state: td && td.getAttribute('data-state'),
        decision: td && td.getAttribute('data-decision'), actual: txt(td && td.querySelector('.st-actual')) }
    })
    const clipped = [...document.querySelectorAll('[data-testid="station-panel"] .st-mcell-top, [data-testid="station-panel"] .st-chip, [data-testid="station-panel"] .st-bend-label')]
      .filter(el => el.scrollWidth > el.clientWidth + 1).map(el => (el.closest('[data-testid^="cell-"], [data-testid="bend-row"]') || el).getAttribute('data-testid'))
    const banner = document.querySelector('[data-testid="summary-banner"]')
    const strip = document.querySelector('[data-testid="stage-strip"]')
    const tags = [...document.querySelectorAll('[data-testid="station-labels"] .st-label')].map(el => ({ uid: el.getAttribute('data-bend-uid'), label: el.getAttribute('data-label'), verdict: el.getAttribute('data-verdict') }))
    // Zero-shot overlay panel ("Detections (unverified)"): line ids, kinds, decisions, review reasons.
    const zsEl = document.querySelector('[data-testid="zs-overlay"]')
    const zs = zsEl ? {
      present: true, lowered: zsEl.getAttribute('data-verdict-lowered') === 'true', effective: zsEl.getAttribute('data-effective-state'),
      lines: [...zsEl.querySelectorAll('[data-testid="zs-line"]')].map(el => ({ id: el.getAttribute('data-line-id'), kind: el.getAttribute('data-kind'),
        decision: el.getAttribute('data-decision'), bend_uid: el.getAttribute('data-bend-uid') || '' })),
      reasons: [...zsEl.querySelectorAll('[data-testid="zs-review-reason"]')].map(el => el.getAttribute('data-code')),
      green: [...zsEl.querySelectorAll('*')].some(el => /rgb\(22, 163, 74\)|rgb\(74, 222, 128\)/.test(getComputedStyle(el).color + getComputedStyle(el).borderColor)),
    } : { present: false }
    return { banner_state: banner && banner.getAttribute('data-summary-state'), banner_verdict: banner && banner.getAttribute('data-verdict'), banner_text: banner && banner.innerText,
      stage_strip: strip && strip.innerText, rows, tags, general_measurements: gms, clipped, zs }
  })
}

/** The open callout: bend, verdict, lines (kind/side/decision) and whether its leader is drawn. */
async function calloutDom(p) {
  return p.evaluate(() => {
    const card = document.querySelector('[data-testid="bend-callout"]')
    if (!card) return null
    const line = document.querySelector('.st-co-leader')
    return { uid: card.getAttribute('data-bend-uid'), label: card.getAttribute('data-label'), verdict: card.getAttribute('data-verdict'),
      visible: getComputedStyle(card).visibility !== 'hidden', placement: card.dataset.placement,
      leader: !!line && line.getAttribute('visibility') !== 'hidden',
      lines: [...card.querySelectorAll('.st-co-line')].map(l => ({ kind: l.dataset.kind, side: l.dataset.side || null, decision: l.dataset.decision })),
      text: card.innerText }
  })
}

const scanJobs = {}
async function scanFlow(key, n) {
  const s = REAL.scans[key]
  const dir = join(OUT, 'scans', key)
  await mkdir(join(dir, 'pdf'), { recursive: true })
  const info = { key, zeiss_name: s.zeiss, kind: s.kind }
  current.scan = info
  // 1. ZEISS drop into the real hot folder.
  const sha = await dropFile(join(CUSTODY, `${key}.ply`), s.zeiss)
  info.scan_sha256 = sha
  const inProgress = await waitFor(async () => (await station.locator('[data-testid="status-ribbon"][data-ribbon="active"]').count()) ? true : null, 120000, 'the active-job ribbon', 100).catch(() => false)
  if (inProgress) await shot(station, `${n}-${key}-in-progress`, 'station shows the running job (active ribbon)')
  note(inProgress, 'station showed the in-progress state', inProgress)
  const job = await jobForSha(sha)
  info.job_id = job.job_id
  const t0 = Date.now()
  const done = await jobDone(job.job_id)
  info.job = { execution_status: done.execution_status, attempts: done.attempts, wall_s: Math.round((Date.now() - t0) / 1000), failure: done.failure || null, scan_file_name: done.scan_file_name }
  check(done.execution_status === 'completed', `${key}: the job completed`, info.job)
  await station.locator(`[data-testid="station-job-id"][data-job-id="${job.job_id}"]`).waitFor({ timeout: 15000 })
  info.update_lag_s = (Date.now() - Date.parse(done.finished_at)) / 1000
  check(info.update_lag_s <= 5.0, `${key}: station shows the job within 5 s, no reload`, info.update_lag_s)
  // 2. Engine provenance from the job workspace.
  const ws = join(DATA, 'jobs', job.job_id)
  const engineLog = readFileSync(join(ws, 'engine', 'engine.log'), 'utf8')
  info.engine_log_head = engineLog.split('\n').slice(0, 6)
  const res = (await api('GET', `/api/line/jobs/${job.job_id}/result`)).json
  await writeFile(join(dir, 'api_result.json'), JSON.stringify(res, null, 1))
  info.engine = res.engine
  check(res.engine && res.engine.engine_name === 'sherman_production_engine' && /^e3-/.test(res.engine.engine_version), `${key}: the result was produced by the E3 production engine`, res.engine && `${res.engine.engine_name} ${res.engine.engine_version}`)
  info.summary = { validity: res.inspection_validity.state, stage: res.stage_decision.status, inferred_stage: res.stage_decision.inferred_stage, summary: res.summary.state }
  info.bends = res.bends.map(b => ({ label: b.label, seq: b.sequence_index, state: b.state }))
  // 3. Station screen (v2: heatmap off by default, bend tags + list, callout on selection).
  await waitFor(async () => (await sattr('[data-testid="station-viewer"]', 'data-scan-colour')) === 'neutral', 30000, 'scan drawn (heatmap off by default)').catch(() => null)
  await sleep(1500)
  // Zero-shot overlay (job-level field): job output -> station API -> screen (the panel loads after the result).
  const ovApi = await api('GET', `/api/line/jobs/${job.job_id}/zeroshot-overlay`)
  info.overlay_api_status = ovApi.status
  if (ovApi.json) await writeFile(join(dir, 'api_overlay.json'), JSON.stringify(ovApi.json, null, 1))
  check(ovApi.status === 200 && ovApi.json && ovApi.json.schema === 'sherman_zeroshot_overlay_v1', `${key}: the job carries a zero-shot overlay (GET /zeroshot-overlay 200)`, ovApi.status)
  await station.locator('[data-testid="zs-overlay"]').waitFor({ timeout: 15000 })
  const px = await canvasPixels(station)
  info.canvas = px
  check(px.painted > 1000 && px.colours > 20, `${key}: WebGL canvas shows the part surface`, px)
  const dom = await stationDom(station)
  await writeFile(join(dir, 'station_dom.json'), JSON.stringify(dom, null, 1))
  check(dom.rows.length === nBends, `${key}: one list row per recipe bend`, dom.rows.length)
  const domLabelMiss = dom.rows.filter(r => r.label !== studioLabelBySeq[r.seq]).map(r => `${r.seq}:${r.label}!=${studioLabelBySeq[r.seq]}`)
  check(domLabelMiss.length === 0, `${key}: station table bend labels = the studio labels (e.g. sequence 6 = ${studioLabelBySeq[6]})`, domLabelMiss)
  const ctxLabels = (res.context && res.context.bend_display_labels) || {}
  check(res.bends.every(b => ctxLabels[b.bend_uid] === studioLabelBySeq[b.sequence_index]), `${key}: result.context.bend_display_labels = the studio labels`, ctxLabels)
  check(dom.clipped.length === 0, `${key}: no clipped text in the bend list`, dom.clipped)
  const tagMiss = res.bends.filter(b => !dom.tags.some(tg => tg.uid === b.bend_uid && tg.label === ctxLabels[b.bend_uid])).map(b => `${b.sequence_index}:${ctxLabels[b.bend_uid]}`)
  check(tagMiss.length === 0, `${key}: every bend has a 3D tag with its studio label`, tagMiss)
  // The overlay may only lower a pass-like E3 state to REVIEW_REQUIRED (never creates or upgrades a pass).
  const shownState = ovApi.json && ovApi.json.verdict_lowered ? ovApi.json.effective_summary_state : res.summary.state
  const verdictWant = { OUT_OF_TOLERANCE_FOUND: 'FIX', REVIEW_REQUIRED: 'CHECK', ALL_MEASURED_IN_TOLERANCE: 'OK', STAGE_IN_TOLERANCE: 'OK_SO_FAR', INCOMPLETE: 'UNSURE', INVALID_INSPECTION: 'RESCAN' }[shownState]
  check(dom.banner_verdict === verdictWant, `${key}: top banner verdict ${dom.banner_verdict} = plan mapping of ${shownState} (E3 ${res.summary.state}${shownState !== res.summary.state ? ', lowered by the overlay' : ''})`, dom.banner_verdict)
  check(dom.zs && dom.zs.present && !dom.zs.green, `${key}: overlay detections shown apart from the bends, never green`, dom.zs)
  check(new RegExp(`${res.summary.bends_formed}\\s*/\\s*${res.summary.bends_total}`).test(dom.stage_strip || ''), `${key}: banner shows ${res.summary.bends_formed} / ${res.summary.bends_total} bends done`, dom.stage_strip)
  info.heatmap_legend = await station.locator('[data-testid="heatmap-legend"]').innerText().catch(() => null)
  const lp = labelProblems(await station.evaluate(probeLabels), { minVisible: 1 })
  note(lp.length === 0, `${key}: bend labels clean`, lp)
  await settle(station)
  await shot(station, `${n}-${key}-station`, `${key}: banner ${dom.banner_state} (${dom.banner_verdict}); ${dom.rows.length} rows; heatmap off`)
  // The heatmap stays one tap away: on, it draws the engine's deviations; off again for the bender view.
  await station.click('[data-layer="heatmap"]')
  const heatOn = await waitFor(async () => (await sattr('[data-testid="station-viewer"]', 'data-scan-colour')) === 'heatmap' ? true : null, 15000, 'heatmap on').catch(() => false)
  if (res.viewer_assets.deviations) check(heatOn === true, `${key}: the heatmap toggle colours the scan by the engine deviations`, await sattr('[data-testid="station-viewer"]', 'data-scan-colour'))
  await sleep(600)
  await shot(station, `${n}-${key}-heatmap`, `${key}: heatmap layer on`)
  await station.click('[data-layer="heatmap"]')
  await waitFor(async () => (await sattr('[data-testid="station-viewer"]', 'data-scan-colour')) === 'neutral' ? true : null, 8000, 'heatmap off').catch(() => null)
  // Worst first: selects the worst bend, focuses it and opens its callout on a leader line.
  await station.click('[data-action="worst"]').catch(() => {})
  await sleep(1400)
  const selUid = await sattr('[data-testid="station-viewer"]', 'data-selected-bend')
  const co = await calloutDom(station)
  info.callout = co && { ...co, text: undefined }
  if (selUid) {
    check(!!co && co.uid === selUid && co.visible && co.leader, `${key}: the selected bend's callout is shown on a leader line`, info.callout)
    check(co && co.label === ctxLabels[selUid], `${key}: the callout names the studio label (${ctxLabels[selUid]})`, co && co.label)
    const selBend = res.bends.find(b => b.bend_uid === selUid)
    const want = selBend.characteristics.slice(0, 3).map(c => `${c.kind}/${c.side || ''}/${c.decision}`)
    const got = (co ? co.lines : []).slice(0, 3).map(l => `${l.kind}/${l.side || ''}/${l.decision}`)
    check(JSON.stringify(got) === JSON.stringify(want), `${key}: callout lines = flange A, flange B, angle with the engine decisions`, { want, got })
    const lp2 = labelProblems(await station.evaluate(probeLabels), { selectedUid: selUid, minVisible: 1 })
    note(lp2.length === 0, `${key}: tags clean with the callout open`, lp2)
  }
  await shot(station, `${n}-${key}-worst-callout`, `worst first: ${co ? `${co.label} ${co.verdict}` : 'no callout'}`)
  await station.click('[data-action="fit"]').catch(() => {})
  await sleep(400)
  // Scroll the list so the later rows (stage scans: the greyed not-yet-formed bends) are in view.
  await station.locator('[data-testid="bend-row"]').last().scrollIntoViewIfNeeded().catch(() => {})
  await sleep(400)
  await shot(station, `${n}-${key}-list-end`, 'the end of the bend list')
  await station.locator('[data-testid="bend-row"]').first().scrollIntoViewIfNeeded().catch(() => {})
  // 4. PDF with the station's own button.
  const [download] = await Promise.all([station.waitForEvent('download', { timeout: 60000 }), station.click('[data-testid="download-pdf"]')])
  const pdfPath = join(dir, 'pdf', 'report.pdf')
  await download.saveAs(pdfPath)
  const pr = run(PY, [SUPPORT, 'pdf-pages', '--pdf', pdfPath, '--out', join(dir, 'pdf'), '--expect', realSku, '--header-ids', realSku, recipeId], { cwd: REPO, env: repoEnv })
  info.pdf = lastJson(pr.out)
  check(pr.code === 0 && info.pdf.ok, `${key}: PDF rendered to page images; header ids in order`, info.pdf.header)
  const pb = run(PY, [RSUPPORT, 'pdf-bends', '--pdf', pdfPath], { cwd: REPO, env: repoEnv })
  await writeFile(join(dir, 'pdf_bends.json'), pb.out.trim().split('\n').pop() || '{}')
  check(pb.code === 0, `${key}: PDF per-bend tables parsed`, pb.err.slice(-300))
  {
    const pdfBends = (JSON.parse(pb.out.trim().split('\n').pop() || '{}').bends) || {}
    const studio = new Set(Object.values(studioLabelBySeq))
    const formedSeqs = res.bends.filter(b => b.state !== 'NOT_YET_FORMED').map(b => studioLabelBySeq[b.sequence_index])
    const pdfLabels = Object.keys(pdfBends)
    check(pdfLabels.every(l => studio.has(l)) && formedSeqs.every(l => pdfLabels.includes(l)), `${key}: PDF bend labels = the studio labels`, pdfLabels)
  }
  // 4b. Overlay equality: job output == station API == screen == PDF "detections (unverified)" section.
  const ovc = run(PY, [RSUPPORT, 'overlay', '--job-dir', ws, '--api', join(dir, 'api_overlay.json'), '--dom', join(dir, 'station_dom.json'), '--pdf', pdfPath], { cwd: REPO, env: repoEnv })
  info.overlay = lastJson(ovc.out)
  await writeFile(join(dir, 'overlay_compare.json'), JSON.stringify(info.overlay, null, 1))
  check(ovc.code === 0, `${key}: zero-shot overlay identical in job output, station API, screen and PDF`, info.overlay.problems || ovc.err.slice(-300))
  // 5. Direct E3 CLI on the same compiled recipe and the same PLY (the custody copy), same bundle.
  const bundleDir = join(DATA, 'engine', 'current')
  // Direct outputs go to the temp work dir (the registered real scan never lands in docs/); result.json + log are copied.
  const directOut = join(WORK, 'direct', key, 'engine')
  const cmd = ['-m', 'domains.production.engine.inspect', '--compiled', join(ws, 'compiled', 'compiled_part.json'), '--recipe', join(ws, 'recipe.json'),
    '--scan', join(CUSTODY, `${key}.ply`), '--out', directOut, '--scan-sha256', sha,
    '--compiled-sha256', JSON.parse(readFileSync(join(ws, 'job.json'), 'utf8')).compiled_part_sha256,
    '--recipe-sha256', JSON.parse(readFileSync(join(ws, 'job.json'), 'utf8')).recipe_sha256,
    '--time-budget-s', '170', '--engine-bundle-id', res.engine.bundle_id]
  info.direct_cmd = `cd ${bundleDir} && OMP_NUM_THREADS=1 PYTHONPATH=${bundleDir}:${bundleDir}/backend ${PY} ${cmd.join(' ')}`
  const dr = run(PY, cmd, { cwd: bundleDir, env: { PATH: process.env.PATH, HOME: process.env.HOME, TMPDIR: process.env.TMPDIR || tmpdir(), OMP_NUM_THREADS: '1', PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', PYTHONPATH: `${bundleDir}:${bundleDir}/backend` }, timeout: 300000 })
  await writeFile(join(dir, 'direct_engine.log'), dr.out + dr.err)
  check(dr.code === 0 && existsSync(join(directOut, 'result.json')), `${key}: direct E3 CLI run exit 0`, (dr.err || dr.out).slice(-400))
  const cmp = run(PY, [RSUPPORT, 'compare', '--direct', join(directOut, 'result.json'), '--job-engine', join(ws, 'engine', 'result.json'),
    '--job-rendered', join(ws, 'result', 'shopfloor_result.json'), '--api', join(dir, 'api_result.json'), '--dom', join(dir, 'station_dom.json'),
    '--pdf', join(dir, 'pdf_bends.json'), '--out', dir], { cwd: REPO, env: repoEnv })
  info.compare = lastJson(cmp.out)
  check(cmp.code === 0, `${key}: every per-bend value, state and decision is identical in E3 CLI, job, API, station screen and PDF`, info.compare.problems)
  const hm = run(PY, [RSUPPORT, 'heatmap', '--job-dir', ws, '--direct-engine', directOut], { cwd: REPO, env: repoEnv, timeout: 600000 })
  info.heatmap = lastJson(hm.out)
  check(hm.code === 0, `${key}: station heatmap values are the engine's per-point deviations (and the direct run's)`, info.heatmap)
  if (info.heatmap.available === false) {
    check(/unavailable/i.test(info.heatmap_legend || '') && info.heatmap.viewer_deviations_file_absent, `${key}: no heatmap published (${info.heatmap.reason}); the station says so and draws none`, info.heatmap_legend)
  }
  await mkdir(join(dir, 'direct_engine'), { recursive: true })
  for (const f of ['result.json', 'engine.log']) await copyFile(join(directOut, f), join(dir, 'direct_engine', f)).catch(() => {})
  // Keep a copy of the job's engine outputs (small files) for the audit.
  for (const f of ['result.json', 'engine.log']) await copyFile(join(ws, 'engine', f), join(dir, `job_engine_${f}`)).catch(() => {})
  scanJobs[key] = info
  return info
}

let n = 10
for (const key of SCANS_WANTED) {
  n += 1
  const nn = n
  await step(`R-${key}`, `drop the real ${key} PLY into the hot folder (ZEISS-style); runner -> E3 -> station -> PDF; direct E3 CLI equality`, async () => {
    const info = await scanFlow(key, nn)
    if (REAL.scans[key].kind === 'PARTIAL') {
      const nyf = info.bends.filter(b => b.state === 'NOT_YET_FORMED')
      note(nyf.length > 0, `${key}: engine reports not-yet-formed bends on the stage scan`, info.bends.map(b => `${b.label}:${b.state}`))
      note(!['ALL_MEASURED_IN_TOLERANCE', 'STAGE_IN_TOLERANCE'].includes(info.summary.summary) || nyf.length > 0, `${key}: stage scan not shown as a finished green part`, info.summary)
    }
  }, { requires: ['W4', 'K1'] })
}

await step('R-hebrew', 'station in Hebrew (RTL) on the last real job', async () => {
  await station.click('[data-testid="lang-toggle"]')
  await station.locator('[data-testid="station-root"][dir="rtl"]').waitFor()
  await sleep(600)
  const m = await station.evaluate(() => ({ sw: document.documentElement.scrollWidth, sh: document.documentElement.scrollHeight }))
  check(m.sw <= 1920 && m.sh <= 1080, 'RTL station has no page scroll', m)
  await shot(station, 'station-hebrew-rtl', 'station-root dir=rtl')
  check((await station.locator('.st-banner-title').innerText()).trim() !== '' && !/[A-Z]{2,}/.test(await station.locator('.st-banner-title').innerText()), 'Hebrew verdict word', await station.locator('.st-banner-title').innerText())
  await station.locator('[data-testid="bend-row"]').first().click()
  await sleep(1400)
  const coHe = await calloutDom(station)
  check(!!coHe && coHe.visible && coHe.leader, 'Hebrew: a tapped bend opens its callout on a leader line', coHe && { ...coHe, text: undefined })
  await shot(station, 'station-hebrew-rtl-callout', `Hebrew callout ${coHe && coHe.label}`)
  await station.click('[data-action="fit"]').catch(() => {})
  await station.click('[data-testid="lang-toggle"]')
}, { requires: ['K1'] })

// ------------------------------------------------------------------ station part picker (held-back scans)
// A re-export of an allowed scan: the same bytes with one extra PLY header comment (new sha256, same geometry),
// so the hotfolder never treats it as a duplicate of the scan already inspected.
async function scanVariant(srcKey, tag, outName) {
  const raw = readFileSync(join(CUSTODY, `${srcKey}.ply`))
  const nl = raw.indexOf(10)
  assert.equal(raw.subarray(0, nl).toString('ascii').trim(), 'ply')
  const dst = join(WORK, 'variants', outName)
  await mkdir(dirname(dst), { recursive: true })
  await writeFile(dst, Buffer.concat([raw.subarray(0, nl + 1), Buffer.from(`comment ${tag}\n`, 'ascii'), raw.subarray(nl + 1)]))
  return dst
}
async function eventsOf(kind) {
  const r = await api('GET', `/api/line/events?station_id=${STATION}&limit=1000`)
  return ((r.json && r.json.events) || []).filter(e => e.kind === kind)
}
async function pickerLayout(p) {
  return p.evaluate(() => {
    const dlg = document.querySelector('[data-testid="part-picker"]')
    if (!dlg) return null
    const r = dlg.getBoundingClientRect()
    const overflowing = [...dlg.querySelectorAll('*')].filter(el => {
      const b = el.getBoundingClientRect()
      return b.width > 0 && (b.left < r.left - 1 || b.right > r.right + 1)
    }).map(el => el.className || el.tagName).slice(0, 8)
    return { dir: getComputedStyle(dlg).direction, rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
      overflowing, pageScroll: [document.documentElement.scrollWidth, document.documentElement.scrollHeight] }
  })
}

await step('P1', 'part picker: an unknown-name scan and an unnamed (Hebrew) scan are held back; the operator picks 10839000 at the station (Hebrew/RTL), the job runs E3 and is marked "assigned by operator"; the other is dismissed with a reason', async () => {
  const pk = { unknown_name: '10839 000_01_FULL_SCAN 9.ply', unnamed_name: 'סריקה ללא שם 1.ply' }
  current.picker = pk
  const unknownSrc = await scanVariant('full', 'sherman-e2e picker unknown-name', 'unknown.ply')
  const unnamedSrc = await scanVariant('scan_1', 'sherman-e2e picker unnamed', 'unnamed.ply')
  pk.unknown_sha = await dropFile(unknownSrc, pk.unknown_name)
  const held1 = await waitFor(async () => {
    const r = await api('GET', `/api/line/quarantine/held?station_id=${STATION}`)
    return r.json && r.json.items && r.json.items.length >= 1 ? r.json : null
  }, 120000, 'the unknown-name scan held back')
  pk.unknown_held = held1.items[0]
  check(held1.items[0].file_name === pk.unknown_name && held1.items[0].reason_code === 'UNMATCHED_SKU', 'unknown name held back as UNMATCHED_SKU', { file: held1.items[0].file_name, reason: held1.items[0].reason_code })
  check(held1.items[0].suggestions[0] && held1.items[0].suggestions[0].sku === realSku, 'the filename fuzzy match suggests 10839000 first', held1.items[0].suggestions)
  pk.unnamed_sha = await dropFile(unnamedSrc, pk.unnamed_name)
  const held2 = await waitFor(async () => {
    const r = await api('GET', `/api/line/quarantine/held?station_id=${STATION}`)
    return r.json && r.json.items && r.json.items.length >= 2 ? r.json : null
  }, 120000, 'two scans held back')
  const unknownId = held2.items.find(i => i.file_name === pk.unknown_name).scan_file_id
  const unnamed = held2.items.find(i => i.file_name === pk.unnamed_name)
  check(!!unnamed, 'the Hebrew/space filename survives ingest unchanged', held2.items.map(i => i.file_name))
  pk.unnamed_reason = unnamed.reason_code
  check(['UNMATCHED_SKU', 'UNPARSEABLE_FILENAME'].includes(unnamed.reason_code), 'the unnamed scan is held back (no part number in the name)', unnamed.reason_code)
  // Station: "Scan held back" now offers "Choose part" with the count.
  await station.locator('[data-testid="choose-part"][data-held="2"]').waitFor({ timeout: 30000 })
  pk.ribbon_reason = await sattr('[data-testid="status-ribbon"]', 'data-reason')
  await shot(station, 'picker-ribbon-held', `ribbon: Scan held back (${pk.ribbon_reason}) + Choose part (2)`)
  // Hebrew (RTL) for the whole picker flow.
  await station.click('[data-testid="lang-toggle"]')
  await station.locator('[data-testid="station-root"][dir="rtl"]').waitFor()
  await station.click('[data-testid="choose-part"]')
  await station.locator('[data-testid="part-picker"]').waitFor()
  await station.locator('[data-testid="picker-held-list"]').waitFor()
  check(await station.locator('.st-picker-held').count() === 2, 'the picker lists both held-back scans')
  await station.click(`.st-picker-held[data-scan-id="${unknownId}"]`)
  await station.locator(`[data-testid="picker-rows"] [data-sku="${realSku}"]`).waitFor({ timeout: 15000 })
  const firstRow = station.locator('[data-testid="picker-rows"] [role="radio"]').first()
  check((await firstRow.getAttribute('data-sku')) === realSku && !!(await firstRow.getAttribute('data-suggested')), 'suggested part first in the list', await firstRow.getAttribute('data-sku'))
  check((await station.locator('[data-testid="picker-filename"]').innerText()).trim() === pk.unknown_name, 'the picker shows the file name', await station.locator('[data-testid="picker-filename"]').innerText())
  pk.why_he = (await station.locator('[data-testid="picker-why"]').innerText()).trim()
  check(/[֐-׿]/.test(pk.why_he), 'why-held text is Hebrew', pk.why_he)
  const lay = await pickerLayout(station)
  pk.layout_he = lay
  check(lay.dir === 'rtl' && lay.overflowing.length === 0 && lay.pageScroll[0] <= 1920 && lay.pageScroll[1] <= 1080, 'picker is RTL, nothing spills out of the dialog, no page scroll at 1920x1080', lay)
  await settle(station)
  await shot(station, 'picker-he-suggested', 'picker (Hebrew/RTL): why held, file name, suggested 10839000 first')
  await station.fill('[data-testid="picker-search"]', '1083')
  await sleep(700)
  await station.click(`[data-testid="picker-rows"] [data-sku="${realSku}"]`)
  await station.locator('[data-testid="picker-options"]').waitFor()
  pk.version_selected = await station.locator('[data-testid="picker-version"]').inputValue()
  pk.stage_default = await station.locator('[data-testid="picker-stage"]').inputValue()
  check(pk.version_selected === recipeId, 'the active recipe version is pre-selected', pk.version_selected)
  check(pk.stage_default === 'FINAL', 'stage defaults to the recipe (FINAL)', pk.stage_default)
  await shot(station, 'picker-he-search-select', `search "1083", selected ${realSku}, version ${pk.version_selected}, stage ${pk.stage_default}`)
  await station.click('[data-testid="picker-next"]')
  await station.locator('[data-testid="picker-confirm"]').waitFor()
  pk.confirm = { sku: (await station.locator('[data-testid="confirm-sku"]').innerText()).trim(), version: (await station.locator('[data-testid="confirm-version"]').innerText()).trim(),
    revision: (await station.locator('[data-testid="confirm-revision"]').innerText().catch(() => '')).trim() }
  check(pk.confirm.sku === realSku && pk.confirm.version === `v${recipeId.split('@v')[1]}` && pk.confirm.revision === REAL.revision, 'confirm step shows SKU / revision / version', pk.confirm)
  const layC = await pickerLayout(station)
  check(layC.overflowing.length === 0, 'confirm step: nothing spills out of the dialog', layC)
  await shot(station, 'picker-he-confirm', `confirm: ${pk.confirm.sku} rev ${pk.confirm.revision} ${pk.confirm.version}`)
  // A double click on confirm: the server is idempotent, one assignment only.
  await station.locator('[data-testid="picker-confirm-run"]').dblclick()
  await station.locator('[data-testid="picker-done"][data-result="assigned"]').waitFor({ timeout: 20000 })
  await shot(station, 'picker-he-assigned', 'assigned; the picker moves on to the other held scan')
  // The picker moves to the remaining scan: dismiss it with a reason.
  await station.locator(`[data-testid="picker-scan"]`).waitFor()
  await waitFor(async () => (await station.locator('[data-testid="picker-filename"]').innerText()).trim() === pk.unnamed_name ? true : null, 10000, 'the unnamed scan in the picker')
  await station.click('[data-testid="picker-to-dismiss"]')
  await station.locator('[data-testid="picker-dismiss"]').waitFor()
  await station.check('input[name="dismiss-reason"][value="RESCAN"]')
  await shot(station, 'picker-he-dismiss', 'dismiss with reason: rescan')
  await station.click('[data-testid="picker-dismiss-do"]')
  await station.locator('[data-testid="picker-done"][data-result="dismissed"]').waitFor({ timeout: 20000 })
  await station.locator('[data-testid="part-picker"]').waitFor({ state: 'detached', timeout: 10000 })
  // Audit: one assignment (double click), one dismissal, with operator, time, from-outcome and the chosen recipe.
  const assigned = (await eventsOf('QUARANTINE_ASSIGNED')).filter(e => e.scan_file_id === unknownId)
  const dismissed = (await eventsOf('QUARANTINE_DISMISSED')).filter(e => e.scan_file_id === unnamed.scan_file_id)
  pk.assigned_event = assigned[0] || null
  pk.dismissed_event = dismissed[0] || null
  check(assigned.length === 1, 'exactly one QUARANTINE_ASSIGNED for the double click', assigned.length)
  const ap = assigned[0].payload
  check(ap.from_outcome === 'UNMATCHED' && ap.recipe_id === recipeId && ap.sku === realSku && ap.assigned_at && assigned[0].principal && assigned[0].principal.role === 'admin' && ap.operator_name === ADMIN.username,
    'assign event: operator, time, from-outcome, chosen recipe', { principal: assigned[0].principal, payload: ap })
  check(dismissed.length === 1 && dismissed[0].payload.dismiss_reason === 'RESCAN' && dismissed[0].payload.operator_name === ADMIN.username, 'dismiss event with the reason and operator', dismissed[0] && dismissed[0].payload)
  // The job runs E3 like any other; the verdict is E3's.
  const job = await jobForSha(pk.unknown_sha)
  pk.job_id = job.job_id
  const done = await jobDone(job.job_id)
  pk.job = { execution_status: done.execution_status, failure: done.failure || null }
  check(done.execution_status === 'completed', 'the operator-assigned job completed', pk.job)
  const detail = (await api('GET', `/api/line/jobs/${job.job_id}`)).json
  pk.assignment = detail.assignment
  check(detail.assignment && detail.assignment.by_operator === true && detail.assignment.from_reason === 'UNMATCHED_SKU', 'job record: part assigned by operator', detail.assignment)
  const res = (await api('GET', `/api/line/jobs/${job.job_id}/result`)).json
  pk.result = { validity: res.inspection_validity.state, summary: res.summary.state, stage: res.stage_decision.status, engine: `${res.engine.engine_name} ${res.engine.engine_version}` }
  check(/^e3-/.test(res.engine.engine_version) && done.result_summary && done.result_summary.summary_state === res.summary.state, 'the verdict is E3\'s own result (job summary = result file)', pk.result)
  await station.locator(`[data-testid="station-job-id"][data-job-id="${job.job_id}"]`).waitFor({ timeout: 20000 })
  await station.locator('[data-testid="assigned-by-operator"]').waitFor({ timeout: 15000 })
  pk.badge_he = (await station.locator('[data-testid="assigned-by-operator"]').innerText()).trim()
  check(/[֐-׿]/.test(pk.badge_he), 'station shows "part assigned by operator" (Hebrew)', pk.badge_he)
  check(await station.locator('[data-testid="choose-part"]').count() === 0, 'no held-back scan left on the station')
  await sleep(1200)
  const m = await station.evaluate(() => ({ sw: document.documentElement.scrollWidth, sh: document.documentElement.scrollHeight }))
  check(m.sw <= 1920 && m.sh <= 1080, 'RTL station with the badge has no page scroll', m)
  await shot(station, 'picker-he-result', `job ${job.job_id.slice(5, 13)}: E3 ${pk.result.summary}; badge "${pk.badge_he}"`)
  await station.click('[data-testid="lang-toggle"]')
  await station.locator('[data-testid="station-root"][dir="ltr"]').waitFor()
  await sleep(600)
  await shot(station, 'picker-en-result', 'English: "Part assigned by operator" badge')
  await writeFile(join(OUT, 'picker.json'), JSON.stringify(pk, null, 1))
}, { requires: ['W4', 'K1'] })

// ------------------------------------------------------------------ hotfolder edge cases (--hotfolder)
// The real station flow end to end through the real hotfolder poller of the line service: files land in the
// watched inbox the way a scanner PC writes them (slow growing writes, temp name + rename, with and without a
// .done sidecar), the watcher ingests, the SKU is resolved from the name, E3 + the zs overlay run, the station
// shows the result. Inputs: re-exports (one extra header comment) of the allowed 10839000 scans, geometric
// derivatives of them (mirror, densified, truncated) and a synthetic E2 HAT part (the "other part"). Every
// file lives in this run's temp work dir; the temp data root is the only archive touched.
const HF = { cases: {}, expected: {} }
const HSUP = join(REPO, 'scripts', 'production', 'hotfolder_e2e_support.py')
const PASS_LIKE = new Set(['ALL_MEASURED_IN_TOLERANCE', 'STAGE_IN_TOLERANCE'])
const hfCase = (id, o) => { HF.cases[id] = { id, ...(HF.cases[id] || {}), ...o } }
function hsup(argv, timeout = 1800000) {
  const r = run(PY, [HSUP, ...argv], { cwd: REPO, env: repoEnv, timeout })
  if (r.code !== 0) throw new Error(`hotfolder_e2e_support ${argv[0]} failed: ${r.err.slice(-800)}`)
  return lastJson(r.out)
}
const scanReport = shas => hsup(['scan-report', DATA, ...shas], 120000)
async function waitScan(sha, pred, timeoutMs, what) {
  return waitFor(async () => { const r = scanReport([sha])[sha]; return r && pred(r) ? r : null }, timeoutMs, what, 2000)
}
const shaOf = f => sha256(f)
async function sidecarDrop(src, name, sidecar) {
  const sha = await dropFile(src, name)
  await writeFile(join(INBOX, `${name}.done`), JSON.stringify(sidecar))
  return sha
}
// A scanner that streams straight into the final name, in chunks, with one long stall (an SMB hiccup).
async function slowWrite(src, name, { chunk = 4 << 20, everyMs = 300, stallAtFrac = 0.45, stallMs = 6000 } = {}) {
  const data = readFileSync(src)
  const fh = await open(join(INBOX, name), 'w')
  let stalled = false
  try {
    for (let i = 0; i < data.length; i += chunk) {
      await fh.write(data.subarray(i, Math.min(i + chunk, data.length)))
      await fh.sync()
      if (!stalled && i / data.length >= stallAtFrac) { stalled = true; await sleep(stallMs) } else await sleep(everyMs)
    }
  } finally { await fh.close() }
  return createHash('sha256').update(data).digest('hex')
}
async function allEvents() {
  const out = []
  let after = 0
  for (;;) {
    const r = await api('GET', `/api/line/events?station_id=${STATION}&after_id=${after}&limit=1000`)
    const ev = (r.json && r.json.events) || []
    out.push(...ev)
    if (ev.length < 1000) return out
    after = ev[ev.length - 1].id
  }
}
async function heldNow() { return ((await api('GET', `/api/line/quarantine/held?station_id=${STATION}`)).json || { items: [] }).items }
async function restartService(why) {
  const old = service
  old.kill('SIGKILL')
  await waitFor(async () => old.exitCode !== null || old.signalCode !== null ? true : null, 20000, 'the killed service to exit', 100)
  HF.restart = { why, killed_pid: old.pid, at: new Date().toISOString() }
  await sleep(2000)
  service = start('service-restarted', ['-m', 'apps.production_line.service_main', '--build-bundle'], baseEnv, APP)
  HF.restart.new_pid = service.pid
}
const V = {}
let hfDeclaredStage = null

if (HOTFOLDER) {
  await step('H0', 'hotfolder cases: prepare inputs in the temp work dir (re-exports, mirror, densified, truncated, synthetic other part) and register a second SKU (23553000, never released)', async () => {
    V.slow = await scanVariant('full', 'hf slow-write', 'slow.ply')
    V.sidecar = await scanVariant('scan_2', 'hf sidecar', 'sidecar.ply')
    V.amb = await scanVariant('full', 'hf ambiguous', 'amb.ply')
    V.rev = await scanVariant('full', 'hf revision', 'rev.ply')
    V.notrel = await scanVariant('scan_1', 'hf not-released', 'notrel.ply')
    V.before = await scanVariant('scan_1', 'hf before-bending', 'before.ply')
    V.hebrew = await scanVariant('full', 'hf hebrew', 'hebrew.ply')
    V.sidebad = await scanVariant('scan_1', 'hf sidecar-bad', 'sidebad.ply')
    V.perm = await scanVariant('scan_2', 'hf permission', 'perm.ply')
    V.retired = await scanVariant('scan_2', 'hf retired', 'retired.ply')
    V.mirror = join(WORK, 'variants', 'mirror.ply'); current.mirror = hsup(['mirror', join(CUSTODY, 'full.ply'), V.mirror])
    V.dense = join(WORK, 'variants', 'dense.ply'); current.dense = hsup(['densify', join(CUSTODY, 'full.ply'), V.dense, '--factor', '4', '--jitter', '0.05'])
    V.trunc = join(WORK, 'variants', 'trunc.ply'); current.trunc = hsup(['truncate', join(CUSTODY, 'full.ply'), V.trunc, '--frac', '0.6'])
    V.wrong = join(WORK, 'variants', 'wrong.ply'); current.wrong = hsup(['wrong-part', join(WORK, 'e2'), V.wrong])
    // Not a scan at all (JPEG bytes) under a .ply name; and a tiny ASCII STL under a .ply name (content decides: it IS a mesh).
    V.garbage = join(WORK, 'variants', 'garbage.ply')
    await writeFile(V.garbage, Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]), Buffer.from('JFIF\0', 'latin1'), randomBytes(8192)]))
    V.stl = join(WORK, 'variants', 'tiny_stl.ply')
    await writeFile(V.stl, 'solid tiny\n' + 'facet normal 0 0 1\n outer loop\n  vertex 0 0 0\n  vertex 10 0 0\n  vertex 0 10 0\n endloop\nendfacet\n'.repeat(40) + 'endsolid\n')
    V.empty = join(WORK, 'variants', 'empty.ply'); await writeFile(V.empty, '')
    check(current.dense.vertices > 3000000, 'densified scan exceeds the 3,000,000-point validated envelope', current.dense)
    hfDeclaredStage = (scanJobs.scan_2 && scanJobs.scan_2.summary && scanJobs.scan_2.summary.inferred_stage) ?? null
    current.declared_stage_for_sidecar = hfDeclaredStage
    const p = await api('POST', '/api/production/parts', { part_number: '23553000', part_name: 'Second part (registered, never released)', revision: 'B' })
    check(p.status === 201, 'second SKU 23553000 registered (no recipe)', p.status)
    await sleep(11000) // the service's resolver snapshot is cached for 10 s
  }, { requires: ['W4', 'K1'] })

  await step('H1', 'content and name problems dropped together: ambiguous, revision mismatch, not-released, duplicate, truncated, zero-byte, non-PLY bytes, foreign .txt, sidecar sha mismatch; nothing reaches E3', async () => {
    const drops = {
      H3_ambiguous: ['10839000_01 23553000 FULL_SCAN 1.ply', V.amb],
      H4a_revision: ['10839000_02_FULL_SCAN 1.ply', V.rev],
      H4b_not_released: ['23553000_B_FULL_SCAN 1.ply', V.notrel],
      H7_duplicate: ['10839000_01_FULL_SCAN 1 (1).ply', join(CUSTODY, 'full.ply')],
      H8_truncated: ['10839000_01_FULL_SCAN 4.ply', V.trunc],
      H9_not_ply: ['10839000_01_FULL_SCAN 5.ply', V.garbage],
    }
    const t0 = Date.now()
    for (const [id, [name, src]] of Object.entries(drops)) hfCase(id, { file_name: name, sha256: await dropFile(src, name), dropped_at: new Date().toISOString() })
    await writeFile(join(INBOX, '10839000_01_FULL_SCAN 6.ply'), '') // zero bytes, written in place (no rename)
    hfCase('H8_zero_byte', { file_name: '10839000_01_FULL_SCAN 6.ply', sha256: createHash('sha256').update('').digest('hex') })
    await writeFile(join(INBOX, '10839000_01_FULL_SCAN 1.txt'), 'operator note: not a scan\n')
    hfCase('H9_foreign_txt', { file_name: '10839000_01_FULL_SCAN 1.txt' })
    hfCase('H9_sidecar_mismatch', { file_name: '10839000_01_SCAN_1 8.ply', sha256: await sidecarDrop(V.sidebad, '10839000_01_SCAN_1 8.ply', { sha256: '0'.repeat(64) }) })
    // Name problems are held at once; content problems after the 60 s truncated grace; zero bytes after 120 s.
    for (const id of ['H3_ambiguous', 'H4a_revision', 'H4b_not_released']) {
      const r = await waitScan(HF.cases[id].sha256, s => s.ingest_state === 'quarantined', 120000, `${id} held back`)
      hfCase(id, { held_after_s: Math.round((Date.now() - t0) / 1000), scan: r })
    }
    const dupBefore = HF.cases.H7_duplicate.sha256
    await waitFor(async () => !existsSync(join(INBOX, drops.H7_duplicate[0])) ? true : null, 60000, 'the duplicate drop consumed')
    hfCase('H7_duplicate', { scan: scanReport([dupBefore])[dupBefore] })
    for (const id of ['H8_truncated', 'H9_not_ply', 'H9_sidecar_mismatch', 'H8_zero_byte']) {
      const r = await waitScan(HF.cases[id].sha256, s => s.ingest_state === 'quarantined', 240000, `${id} quarantined`)
      hfCase(id, { held_after_s: Math.round((Date.now() - t0) / 1000), scan: r })
    }
    await sleep(3000)
    const left = (await readdir(INBOX)).filter(x => !x.startsWith('.'))
    current.inbox_left = left
    check(left.length === 1 && left[0] === '10839000_01_FULL_SCAN 1.txt', 'only the foreign .txt is left in the inbox (untouched, never ingested)', left)
    hfCase('H9_foreign_txt', { left_untouched: readFileSync(join(INBOX, left[0]), 'utf8') === 'operator note: not a scan\n' })
    const held = await heldNow()
    current.held = held.map(i => ({ file: i.file_name, reason: i.reason_code, resolvable: i.resolvable, suggestions: i.suggestions.map(s => `${s.sku}:${s.source}`) }))
    const amb = held.find(i => i.file_name === drops.H3_ambiguous[0])
    check(amb && amb.reason_code === 'AMBIGUOUS_SKU' && amb.suggestions.length === 1 && amb.suggestions[0].sku === realSku && amb.suggestions[0].source === 'RESOLVER_CANDIDATE',
      'ambiguous name: held with the released candidate suggested (the unreleased one is not)', amb && amb.suggestions)
    const rev = held.find(i => i.file_name === drops.H4a_revision[0])
    check(rev && rev.reason_code === 'REVISION_MISMATCH' && rev.parsed.revision === '02' && rev.suggestions[0] && rev.suggestions[0].sku === realSku, 'revision mismatch: held, revision 02 read, 10839000 suggested', rev && { parsed: rev.parsed, s: rev.suggestions })
    const nr = held.find(i => i.file_name === drops.H4b_not_released[0])
    check(nr && nr.reason_code === 'PART_NOT_READY' && nr.suggestions.length === 0, 'registered but never released: PART_NOT_READY, no assignable suggestion', nr && nr.suggestions)
    const contentHeld = held.filter(i => !i.resolvable).map(i => i.reason_code).sort()
    check(JSON.stringify(contentHeld) === JSON.stringify(['EMPTY_SCAN', 'PLY_TRUNCATED', 'SIDECAR_INVALID', 'UNSUPPORTED_FORMAT'].sort()) || contentHeld.length === 4,
      'content problems are held and cannot be assigned', contentHeld)
    const dupEv = (HF.cases.H7_duplicate.scan.events || []).filter(e => e.kind === 'SCAN_DUPLICATE')
    check(dupEv.length >= 1 && HF.cases.H7_duplicate.scan.jobs.length === 1, 'duplicate bytes: SCAN_DUPLICATE, still exactly one job for those bytes', { dup_events: dupEv.length, jobs: HF.cases.H7_duplicate.scan.jobs.length })
    const jobs = (await api('GET', `/api/line/jobs?station_id=${STATION}&limit=200`)).json.jobs
    check(!jobs.some(j => j.created_at > new Date(t0).toISOString()), 'none of these files queued an E3 job', jobs.filter(j => j.created_at > new Date(t0).toISOString()).map(j => j.job_id))
    // Station: every held scan in the picker list with its reason (English), then the picker's refusal paths.
    await station.locator('[data-testid="choose-part"]').waitFor({ timeout: 30000 })
    current.ribbon = (await station.locator('[data-testid="status-ribbon"]').innerText()).replace(/\s+/g, ' ').trim()
    await shot(station, 'hf-ribbon-held', `ribbon: ${current.ribbon}`)
    await station.click('[data-testid="choose-part"]')
    await station.locator('[data-testid="picker-held-list"]').waitFor()
    current.picker_list = await station.locator('.st-picker-held').evaluateAll(els => els.map(e => ({ id: e.getAttribute('data-scan-id'), text: e.innerText.replace(/\s+/g, ' ').trim() })))
    for (const c of Object.values(HF.cases)) {
      const row = c.scan && current.picker_list.find(p => p.id === c.scan.scan_file_id)
      if (row) hfCase(c.id, { ui_text: row.text })
    }
    const lay = await pickerLayout(station)
    check(lay.overflowing.length === 0, 'picker with 7 held scans: nothing spills out of the dialog', lay)
    await shot(station, 'hf-picker-held-list', `${current.picker_list.length} held scans listed with their reasons`)
    // A content problem in the picker: dismiss only.
    await station.click(`.st-picker-held[data-scan-id="${HF.cases.H8_truncated.scan.scan_file_id}"]`)
    await station.locator('[data-testid="picker-not-resolvable"]').waitFor()
    await shot(station, 'hf-picker-content-problem', 'truncated PLY: cannot be assigned, re-export and dismiss')
    // H3: the operator picks the part for the ambiguous name in the UI.
    await station.click(`.st-picker-held[data-scan-id="${amb.scan_file_id}"]`)
    await station.locator(`[data-testid="picker-rows"] [data-sku="${realSku}"][data-suggested="RESOLVER_CANDIDATE"]`).waitFor()
    await station.click(`[data-testid="picker-rows"] [data-sku="${realSku}"]`)
    await station.click('[data-testid="picker-next"]')
    await station.locator('[data-testid="picker-confirm"]').waitFor()
    await shot(station, 'hf-picker-ambiguous-confirm', 'ambiguous name: confirm 10839000')
    await station.click('[data-testid="picker-confirm-run"]')
    await station.locator('[data-testid="picker-done"][data-result="assigned"]').waitFor({ timeout: 20000 })
    hfCase('H3_ambiguous', { assigned: true })
    await station.click('[data-testid="picker-close"]')
    await station.locator('[data-testid="part-picker"]').waitFor({ state: 'detached' })
    // Refusals through the API (what the UI would get): content problems and an unreleased part cannot be assigned.
    const refusals = {}
    for (const id of ['H4b_not_released', 'H8_truncated', 'H9_sidecar_mismatch']) {
      const r = await api('POST', `/api/line/quarantine/${HF.cases[id].scan.scan_file_id}/assign`, id === 'H4b_not_released' ? { sku: '23553000' } : { sku: realSku })
      refusals[id] = `${r.status} ${r.json && r.json.detail && r.json.detail.code}`
      hfCase(id, { assign_attempt: refusals[id] })
    }
    check(refusals.H4b_not_released === '409 PART_NOT_READY' && refusals.H8_truncated === '409 NOT_RESOLVABLE' && refusals.H9_sidecar_mismatch === '409 NOT_RESOLVABLE', 'assign refused: unreleased part / content problems', refusals)
    for (const id of ['H4a_revision', 'H4b_not_released', 'H8_truncated', 'H8_zero_byte', 'H9_not_ply', 'H9_sidecar_mismatch']) {
      const r = await api('POST', `/api/line/quarantine/${HF.cases[id].scan.scan_file_id}/dismiss`, { reason: 'RESCAN', note: `e2e ${id}` })
      hfCase(id, { dismissed: r.status })
    }
    await rm(join(INBOX, '10839000_01_FULL_SCAN 1.txt'))
  }, { requires: ['H0'] })

  await step('H2', 'several files at once + a slow growing write + the line service killed (SIGKILL) mid-job and mid-write and restarted: wrong part, mirrored, over-density, BEFORE BENDING, Hebrew name, sidecar with declared stage; one job at a time, nothing lost or doubled, never a false PASS', async () => {
    const batch = {
      H5_wrong_part: ['10839000_01_FULL_SCAN 11.ply', V.wrong],
      H6_mirrored: ['10839000_01_FULL_SCAN 12.ply', V.mirror],
      H10_density: ['10839000_01_FULL_SCAN 13.ply', V.dense],
      H11_before_bending: ['BEFORE BENDING  10839000_01.ply', V.before],
      H14_hebrew_space: ['10839000_01 סריקה מלאה 1.ply', V.hebrew],
      H9b_stl_named_ply: ['10839000_01_FULL_SCAN 15.ply', V.stl],
    }
    const t0 = new Date().toISOString()
    for (const [id, [name, src]] of Object.entries(batch)) hfCase(id, { file_name: name, sha256: await dropFile(src, name) })
    const sc = { sha256: shaOf(V.sidecar), size_bytes: readFileSync(V.sidecar).length }
    if (hfDeclaredStage != null) sc.declared_stage = hfDeclaredStage
    hfCase('H1c_sidecar', { file_name: '10839000_01_SCAN_2 5.ply', sidecar: sc, sha256: await sidecarDrop(V.sidecar, '10839000_01_SCAN_2 5.ply', sc) })
    // Slow write in parallel; the service is killed once a job runs (mid-job) while that write is still growing.
    hfCase('H1b_slow_write', { file_name: '10839000_01_FULL_SCAN 14.ply', sha256: shaOf(V.slow) })
    const slow = slowWrite(V.slow, '10839000_01_FULL_SCAN 14.ply', { chunk: 2 << 20, everyMs: 400, stallMs: 8000 })
    const running = await waitFor(async () => {
      const r = await api('GET', `/api/line/jobs?station_id=${STATION}&status=running&limit=5`)
      return r.json && r.json.jobs && r.json.jobs[0] ? r.json.jobs[0] : null
    }, 300000, 'a running job', 500)
    HF.killed_job = running.job_id
    await restartService('SIGKILL mid-job and mid-write')
    await slow
    HF.slow_write_finished = new Date().toISOString()
    await waitFor(async () => (await api('GET', '/api/line/health/ready', null, '')).json?.ready === true ? true : null, 300000, 'line ready after the restart')
    // Every drop -> exactly one job, all completed.
    const ids = ['H1b_slow_write', 'H1c_sidecar', ...(HF.cases.H3_ambiguous && HF.cases.H3_ambiguous.assigned ? ['H3_ambiguous'] : []), ...Object.keys(batch)]
    const done = await waitFor(async () => {
      const rep = scanReport(ids.map(id => HF.cases[id].sha256))
      const ok = ids.every(id => rep[HF.cases[id].sha256] && rep[HF.cases[id].sha256].jobs.length === 1 && ['completed', 'failed'].includes(rep[HF.cases[id].sha256].jobs[0].execution_status))
      return ok ? rep : null
    }, 3600000, 'all batch jobs finished', 5000)
    for (const id of ids) hfCase(id, { scan: done[HF.cases[id].sha256] })
    check(ids.every(id => done[HF.cases[id].sha256].jobs.length === 1), 'every file produced exactly one job (no loss, no double processing across the restart)', Object.fromEntries(ids.map(id => [id, done[HF.cases[id].sha256].jobs.length])))
    check(ids.every(id => done[HF.cases[id].sha256].jobs[0].execution_status === 'completed'), 'every job completed', Object.fromEntries(ids.map(id => [id, done[HF.cases[id].sha256].jobs[0].execution_status])))
    const slowRow = done[HF.cases.H1b_slow_write.sha256]
    check(slowRow.stored_exists && slowRow.stored_rel_path.startsWith('archive/'), 'slow growing write: ingested once, complete bytes archived (never a partial)', slowRow.stored_rel_path)
    check(!existsSync(join(INBOX, '10839000_01_SCAN_2 5.ply.done')), 'the .done sidecar was consumed with its scan')
    const scRow = done[HF.cases.H1c_sidecar.sha256]
    if (hfDeclaredStage != null) check(scRow.jobs[0].declared_stage && scRow.jobs[0].declared_stage.stage_index === hfDeclaredStage && scRow.jobs[0].declared_stage.source === 'SIDECAR', 'sidecar declared stage reaches the job', scRow.jobs[0].declared_stage)
    // One job at a time: E3 running intervals from the event log never overlap.
    const ev = (await allEvents()).filter(e => e.ts >= t0)
    const open = new Map(); const intervals = []
    for (const e of ev) {
      if (e.kind === 'JOB_STARTED') open.set(e.job_id, e.ts)
      if (['JOB_COMPLETED', 'JOB_FAILED', 'JOB_RECOVERED', 'JOB_CANCELLED'].includes(e.kind) && open.has(e.job_id)) { intervals.push([open.get(e.job_id), e.ts, e.job_id, e.kind]); open.delete(e.job_id) }
    }
    for (const [jid, ts] of open) intervals.push([ts, '9999', jid, 'KILLED_UNRECORDED'])
    intervals.sort()
    const overlaps = intervals.filter((iv, i) => i > 0 && iv[0] < intervals[i - 1][1] && intervals[i - 1][3] !== 'KILLED_UNRECORDED')
    HF.intervals = intervals
    check(overlaps.length === 0, 'one E3 job at a time (no overlapping running intervals)', overlaps)
    const recovered = ev.filter(e => e.kind === 'JOB_RECOVERED' && e.job_id === HF.killed_job)
    HF.killed_job_events = ev.filter(e => e.job_id === HF.killed_job).map(e => `${e.ts} ${e.kind}`)
    note(recovered.length >= 1, 'the job killed with the service was recovered (JOB_RECOVERED) by the restarted service', HF.killed_job_events)
    // Never a false PASS on another part, a mirror image or an over-dense scan.
    for (const id of ['H5_wrong_part', 'H6_mirrored', 'H10_density', 'H9b_stl_named_ply']) {
      const s = done[HF.cases[id].sha256].jobs[0].result_summary
      hfCase(id, { verdict: s })
      check(s && !PASS_LIKE.has(s.summary_state), `${id}: never a PASS`, s)
    }
    const dens = done[HF.cases.H10_density.sha256].jobs[0].result_summary
    check(dens.summary_state === 'INVALID_INSPECTION' && dens.inspection_validity === 'UNSUPPORTED_INPUT', 'over-density: INVALID_INSPECTION (UNSUPPORTED_INPUT)', dens)
    const wrong = done[HF.cases.H5_wrong_part.sha256].jobs[0].result_summary
    note(wrong.inspection_validity === 'WRONG_PART_SUSPECTED', 'wrong part under a right name: E3 says WRONG_PART_SUSPECTED', wrong)
    // Station UI per job (history chips): banner verdict, validity text; zs overlay present.
    for (const id of ids) {
      const jid = done[HF.cases[id].sha256].jobs[0].job_id
      const res = (await api('GET', `/api/line/jobs/${jid}/result`)).json
      const ov = await api('GET', `/api/line/jobs/${jid}/zeroshot-overlay`)
      hfCase(id, { job_id: jid, result: { validity: res.inspection_validity.state, reasons: res.inspection_validity.reasons, summary: res.summary.state, stage: res.stage_decision.status, declared: res.stage_decision.declared_stage, inferred: res.stage_decision.inferred_stage }, overlay_status: ov.status, overlay_effective: ov.json && ov.json.effective_summary_state })
      const chip = station.locator(`[data-testid="history-strip"] [data-job-id="${jid}"]`)
      if (await chip.count()) {
        await chip.click()
        await station.locator(`[data-testid="station-job-id"][data-job-id="${jid}"]`).waitFor({ timeout: 20000 })
        await sleep(1500)
        const b = await station.evaluate(() => { const el = document.querySelector('[data-testid="summary-banner"]'); return el ? { verdict: el.getAttribute('data-verdict'), state: el.getAttribute('data-summary-state'), text: el.innerText.replace(/\s+/g, ' ').trim().slice(0, 200) } : null })
        hfCase(id, { ui: b })
        await shot(station, `hf-${id}`, `${HF.cases[id].file_name}: ${b && b.verdict} (${b && b.state}); E3 ${res.inspection_validity.state}/${res.summary.state}`)
      } else hfCase(id, { ui: 'not in the station history strip (older than the last 10 jobs)' })
    }
    await station.goto(`${base}/station/${STATION}`)
    await station.locator('[data-testid="station-root"]').waitFor()
  }, { requires: ['H0'] })

  await step('H3', 'archive permission denied (temp data root only): the file stays in the inbox, the poller backs off and retries; once fixed it is ingested exactly once', async () => {
    const now = new Date()
    const month = join(DATA, 'archive', 'scans', String(now.getUTCFullYear()), String(now.getUTCMonth() + 1).padStart(2, '0'))
    check(existsSync(month), 'archive month dir exists in the temp data root', month)
    const name = '10839000_02_SCAN_2 3.ply'
    await chmod(month, 0o555)
    let sha
    try {
      sha = await dropFile(V.perm, name)
      hfCase('H15_permission_denied', { file_name: name, sha256: sha })
      await sleep(20000)
      const during = scanReport([sha])[sha]
      hfCase('H15_permission_denied', { during: { in_inbox: existsSync(join(INBOX, name)), row: during && { state: during.ingest_state, stored_exists: during.stored_exists } } })
      check(existsSync(join(INBOX, name)), 'while the archive is read-only the scan stays in the inbox (never lost)')
    } finally {
      await chmod(month, 0o755)
    }
    const r = await waitScan(sha, s => s.ingest_state === 'quarantined' && s.stored_exists, 240000, 'the scan ingested after the fix')
    hfCase('H15_permission_denied', { scan: r, inbox_after: existsSync(join(INBOX, name)) })
    check(!existsSync(join(INBOX, name)) && r.stored_exists && r.quarantine_reason === 'REVISION_MISMATCH' && r.jobs.length === 0, 'after the fix: ingested once, archived, held as REVISION_MISMATCH', { state: r.ingest_state, reason: r.quarantine_reason })
    check(r.events.filter(e => e.kind === 'SCAN_RECEIVED').length === 1, 'exactly one SCAN_RECEIVED for it', r.events.map(e => e.kind))
    await api('POST', `/api/line/quarantine/${r.scan_file_id}/dismiss`, { reason: 'RESCAN' })
  }, { requires: ['H0'] })

  await step('H4', 'a retired part: its scans are held (PART_NOT_READY), the picker cannot offer it and assign is refused', async () => {
    const ret = await api('POST', `/api/production/recipes/${encodeURIComponent(recipeId)}/retire`, {})
    check(ret.status === 200, 'recipe retired (admin)', ret.status)
    await sleep(11000)
    const name = '10839000_01_SCAN_2 7.ply'
    const sha = await dropFile(V.retired, name)
    hfCase('H4c_retired', { file_name: name, sha256: sha })
    const r = await waitScan(sha, s => s.ingest_state === 'quarantined', 120000, 'the retired-part scan held back')
    const parts = (await api('GET', `/api/line/released-parts?station_id=${STATION}&q=${realSku}`)).json.items
    const a = await api('POST', `/api/line/quarantine/${r.scan_file_id}/assign`, { sku: realSku })
    const a2 = await api('POST', `/api/line/quarantine/${r.scan_file_id}/assign`, { recipe_id: recipeId })
    hfCase('H4c_retired', { scan: scanReport([sha])[sha], released_parts: parts.map(p => p.sku), assign_attempt: `${a.status} ${a.json && a.json.detail && a.json.detail.code}; by recipe_id: ${a2.status} ${a2.json && a2.json.detail && a2.json.detail.code}` })
    check(r.quarantine_reason === 'PART_NOT_READY' && parts.length === 0 && a.status === 409 && a2.status === 409, 'retired: held PART_NOT_READY, not searchable, assign refused', HF.cases.H4c_retired.assign_attempt)
    const held = await heldNow()
    const item = held.find(i => i.scan_file_id === r.scan_file_id)
    check(item && item.suggestions.length === 0, 'no suggestion for a retired part', item && item.suggestions)
    await station.locator('[data-testid="choose-part"]').waitFor({ timeout: 30000 })
    await station.click('[data-testid="choose-part"]')
    await station.locator('[data-testid="part-picker"]').waitFor()
    await sleep(1200)
    hfCase('H4c_retired', { ui_rows: await station.locator('[data-testid="picker-rows"] [role="radio"]').count() })
    await shot(station, 'hf-picker-retired', 'retired part: nothing to pick')
    await station.click('[data-testid="picker-to-dismiss"]')
    await station.check('input[name="dismiss-reason"][value="OTHER"]')
    await station.fill('.st-picker-notefield textarea', 'recipe retired; engineering to release a new revision')
    await station.click('[data-testid="picker-dismiss-do"]')
    await station.locator('[data-testid="part-picker"]').waitFor({ state: 'detached', timeout: 10000 })
    hfCase('H4c_retired', { scan: scanReport([sha])[sha] })
  }, { requires: ['H0'] })

  // Expected vs actual for every case (HOTFOLDER_E2E.md is written from this file).
  await step('H5', 'hotfolder cases: expected vs actual table, zero false PASS', async () => {
    const fullJob = scanJobs.full || {}
    hfCase('H1a_clean_name', { file_name: REAL.scans.full.zeiss, job_id: fullJob.job_id, result: fullJob.summary, note: 'R-full step of this run (temp name + rename)' })
    hfCase('H2_unknown_pick', { file_name: '10839 000_01_FULL_SCAN 9.ply', note: 'P1 step of this run (picker.json)' })
    const allJobs = Object.values(HF.cases).filter(c => c.scan && c.scan.jobs && c.scan.jobs[0])
    await rm(join(INBOX, '10839000_01_FULL_SCAN 1.txt'), { force: true })
    const falsePass = allJobs.filter(c => ['H5_wrong_part', 'H6_mirrored', 'H10_density', 'H9b_stl_named_ply'].includes(c.id) && PASS_LIKE.has(c.scan.jobs[0].result_summary && c.scan.jobs[0].result_summary.summary_state))
    HF.zero_false_pass = falsePass.length === 0
    check(falsePass.length === 0, 'zero false PASS', falsePass.map(c => c.id))
    await writeFile(join(OUT, 'hotfolder.json'), JSON.stringify(HF, null, 1))
  }, { requires: ['H0'] })
}

await step('Z1', 'no page errors; no legacy worker process; inbox emptied', async () => {
  current.page_errors = pageErrors.slice(0, 20)
  check(pageErrors.length === 0, 'no uncaught page errors', pageErrors.slice(0, 5))
  const logs = (server ? server.logText() : '') + (service ? service.logText() : '')
  check(!/bend_analysis_worker/.test(logs), 'server/service logs never mention the legacy bend_analysis_worker')
  // Negative probe: the discarded legacy runtime is not reachable on the line, even for the admin.
  const legacy = {}
  for (const path of ['/api/analyze', '/api/batch/analyze', '/api/bend-inspection/analyze', '/api/bend-inspection/analyze-from-catalog']) {
    const r = await api('POST', path, {})
    legacy[path] = `${r.status} ${r.json && r.json.code}`
  }
  current.legacy_routes = legacy
  check(Object.values(legacy).every(v => v.startsWith('403 ROUTE_DISABLED')), 'legacy engine routes refused on the line (403 ROUTE_DISABLED)', legacy)
  // The legacy analysis pages are hidden from the sidebar and their routes redirect home while the line is enabled.
  const feat = (await api('GET', '/api/line/features')).json
  current.line_features = feat
  check(feat && feat.line_enabled === true && feat.legacy_engine_enabled === false, 'GET /api/line/features: line enabled, legacy engine off', feat)
  await page.goto(`${base}/jobs`) // '/' on the API port is the API root; any SPA route loads the app shell
  const SIDEBAR = 'nav[aria-label="Main navigation"]'
  await page.locator(`${SIDEBAR} a[href="/jobs"]`).waitFor({ timeout: 15000 })
  await sleep(1500) // the features query settles; legacy entries must not appear afterwards
  const nav = await page.locator(`${SIDEBAR} a`).evaluateAll(els => els.map(e => e.getAttribute('href')))
  current.sidebar = nav
  check(!nav.some(h => ['/upload', '/batch', '/bend-inspection'].includes(h)), 'sidebar hides New Analysis / Batch Processing / Bend Inspection on the line', nav)
  const redirects = {}
  for (const path of ['/upload', '/batch', '/bend-inspection']) {
    await page.goto(`${base}${path}`)
    await page.waitForURL(u => new URL(u).pathname === '/', { timeout: 15000 }).catch(() => {})
    redirects[path] = new URL(page.url()).pathname
  }
  current.legacy_page_redirects = redirects
  check(Object.values(redirects).every(v => v === '/'), 'legacy page routes redirect home on the line', redirects)
  await shot(page, 'sidebar-line', 'sidebar on the line: no legacy analysis pages')
  const left = (await readdir(INBOX)).filter(x => !x.startsWith('.'))
  check(left.length === 0, 'inbox emptied', left)
})

// ------------------------------------------------------------------ teardown
await writeFile(join(OUT, 'scan_jobs.json'), JSON.stringify(scanJobs, null, 1))
try { await browser.close() } catch { /* ignore */ }
for (const c of children) { try { if (c.exitCode === null) c.kill('SIGTERM') } catch { /* ignore */ } }
await sleep(3000)
for (const c of children) { try { if (c.exitCode === null) c.kill('SIGKILL') } catch { /* ignore */ } }
await sleep(500)
for (const c of children) { try { await writeFile(join(OUT, 'logs', `${c.name}.log`), c.logText()) } catch { /* ignore */ } }
const counts = steps.reduce((acc, s) => ({ ...acc, [s.status]: (acc[s.status] || 0) + 1 }), {})
await writeFile(join(OUT, 'steps.json'), JSON.stringify({ run: RUN, counts, steps }, null, 2))
console.log(`\nsteps: ${JSON.stringify(counts)}\nresults: ${join(OUT, 'steps.json')}`)
if (!KEEP) await rm(WORK, { recursive: true, force: true })
else console.log(`kept ${WORK}`)
process.exit(steps.some(s => s.status !== 'PASS') ? 1 : 0)
