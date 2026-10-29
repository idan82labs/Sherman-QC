#!/usr/bin/env node
// App end-to-end run of the Sherman QC production line (plan §6 + §10 amendment 10; C11).
//
// Everything runs on real local servers started from a TEMPORARY APP COPY (C04
// `engine_bundle build --profile app`), never from the dev repo, with a temporary
// SHERMAN_LINE_DATA_ROOT and station inbox. Playwright drives Chromium at 1920x1080.
// Every screenshot is paired with a DOM or API assertion; each step is recorded as
//   PASS | FAIL | NOT_RUN (custody or a prerequisite failed).
// The Bends & order step (wizard step 5) is driven THROUGH THE UI: the wizard opens the
// Vendor Setup tagging studio, the run clicks the auto-marked CAD bends to confirm them,
// tags the order and one מהלך group of two bends, toggles folded/unfolded, saves, returns
// to the wizard and only then asserts through the API. The old API assist (PUT
// vendor-setup) survives only as a fallback when the UI path breaks; the step is then
// FAIL (the fallback only lets the later steps run, never a pass).
//
// Output: docs/production/runlog/e2e/<run>/ with NN-*.png screenshots, steps.json
// (assertions per step), pdf/ page renders, logs/ (server + service logs) and a
// summary line. The visual review of the screenshots happens in wave 4 (V1).
//
// Data: an E2 part set (HAT, DEV seeds, a real STEP BRep, its flat DXF, a drawing PDF and
// scans of the same part) is generated into the temp directory. The real development part
// 10839000 (Gold11 dev part, authorized under plan §9) is onboarded the same way from its
// STEP in ../source-assets/02_CAD_QC_3D/ and its FULL scan is dropped into the station
// inbox; both are read-only copies inside the temp directory. It has no authorized
// drawing PDF, so a clearly labelled placeholder PDF is attached (the wizard requires one).
// Run the whole driver under the shared geometry lock (it runs engine jobs).
//
// Usage: node scripts/production/run_e2e_app.mjs [--headed] [--keep] [--run NAME]
//   --keep  leave the temp directory (app copy, data root) in place for inspection
// Exit: 0 when every executed step is PASS, 1 otherwise, 2 setup failure.
import assert from 'node:assert/strict'
import { spawn, spawnSync } from 'node:child_process'
import { createHash, randomBytes } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { chmod, copyFile, mkdir, mkdtemp, readdir, rename, rm, utimes, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import net from 'node:net'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const FE = join(REPO, 'frontend', 'react')
// Station bend-label probe (shared with check-station-layout.mjs): overlap / mis-tag / covered-control checks.
const { probeLabels, labelProblems } = await import(join(FE, 'scripts', 'station-label-probe.mjs'))
const PY = join(REPO, '.venv', 'bin', 'python')
const SUPPORT = join(REPO, 'scripts', 'production', 'run_e2e_app_support.py')
const args = process.argv.slice(2)
const HEADED = args.includes('--headed')
const KEEP = args.includes('--keep')
const RUN = args.includes('--run') ? args[args.indexOf('--run') + 1] : new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z')
const OUT = join(REPO, 'docs', 'production', 'runlog', 'e2e', RUN)
const STATION = 'cell-1'
const REAL = { stem: '10839000_01', part_number: '10839000', revision: '01', part_name: 'Gold11 dev part 10839000',
  step: resolve(REPO, '..', 'source-assets', '02_CAD_QC_3D', '10839000_01.stp'),
  fullScan: resolve(REPO, '..', 'source-assets', '02_CAD_QC_3D', '10839000_01_full_scan.ply'),
  zeiss: '10839000_01_FULL_SCAN 1.ply' }
const ADMIN = { username: 'e2e-admin', password: `E2e-${randomBytes(9).toString('base64url')}` }
const sleep = ms => new Promise(r => setTimeout(r, ms))

let chromium
try {
  ;({ chromium } = createRequire(join(FE, 'package.json'))('playwright'))
  if (!existsSync(chromium.executablePath())) throw new Error(`no Chromium at ${chromium.executablePath()}`)
} catch (err) {
  console.error(`run_e2e_app: Playwright Chromium not available (${err.message})`)
  process.exit(2)
}

// ------------------------------------------------------------------ bookkeeping
const steps = []
const failurePages = []
let shotNo = 0
let current = null
function record(status, detail) { current.status = status; if (detail) current.detail = String(detail).slice(0, 2000) }
async function step(id, title, fn, { requires = [] } = {}) {
  current = { id, title, status: 'PASS', assertions: [], screenshots: [], started: new Date().toISOString() }
  steps.push(current)
  const t0 = Date.now()
  // A FAIL whose API fallback succeeded (`unblocks`) lets later steps run; it is still a FAIL.
  const missing = requires.filter(r => !steps.find(s => s.id === r && (s.status === 'PASS' || s.unblocks)))
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
  console.log(`${current.status.padEnd(10)} ${id} ${title}${current.detail && current.status !== 'PASS' ? `\n           ${current.detail.slice(0, 300)}` : ''}`)
  await writeFile(join(OUT, 'steps.json'), JSON.stringify({ run: RUN, steps }, null, 2))
}
function check(cond, what, value) {
  current.assertions.push({ what, ok: !!cond, value: value === undefined ? null : value })
  assert.ok(cond, `${what}${value === undefined ? '' : ` (${JSON.stringify(value).slice(0, 300)})`}`)
}
/** Informational assertion: recorded with its outcome, never fails the step. */
function note(cond, what, value) {
  current.assertions.push({ what, ok: !!cond, value: value === undefined ? null : value, informational: true })
}
/**
 * Wait until the page has settled before an evidence screenshot: no running finite
 * animations or transitions (banner fade-in, toggle colour), no loading wizard.
 */
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
  const r = spawnSync(cmd, argv, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, ...opts })
  return { code: r.status, out: r.stdout || '', err: r.stderr || '' }
}
function start(name, argv, env, cwd) {
  const logPath = join(OUT, 'logs', `${name}.log`)
  const child = spawn(PY, argv, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'], detached: false })
  const chunks = []
  const onData = d => { chunks.push(d); }
  child.stdout.on('data', onData)
  child.stderr.on('data', onData)
  child.on('exit', async () => { try { await writeFile(logPath, Buffer.concat(chunks)) } catch { /* ignore */ } })
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

// ------------------------------------------------------------------ setup
await mkdir(join(OUT, 'logs'), { recursive: true })
await mkdir(join(OUT, 'pdf'), { recursive: true })
const WORK = await mkdtemp(join(process.env.TMPDIR || tmpdir(), 'sherman-e2e-app-'))
const APP = join(WORK, 'app')
const DATA = join(WORK, 'line')
const INBOX = join(WORK, 'inbox')
await mkdir(INBOX, { recursive: true })
console.log(`run ${RUN}\n  out  ${OUT}\n  work ${WORK}`)

let base = ''
let token = ''
let partSet = null
let setDir = ''
let server = null
let service = null
let partId = ''
let recipeId = ''
let serviceEnv = null
let browser = null
const api = async (method, path, body, tok = token) => {
  const r = await fetch(`${base}${path}`, {
    method, headers: { ...(tok ? { Authorization: `Bearer ${tok}` } : {}), ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  })
  const text = await r.text()
  let json = null
  try { json = JSON.parse(text) } catch { /* not json */ }
  return { status: r.status, json, text, headers: r.headers }
}
const scan = name => partSet.scans.find(s => s.name === name)
async function dropScan(name, fileName) {
  return dropFile(join(setDir, scan(name).dir, 'scan.ply'), fileName)
}
// ZEISS-like drop: write to a hidden temp name in the inbox, rename, age the mtime.
async function dropFile(src, fileName) {
  const tmp = join(INBOX, `.${fileName}.part`)
  await copyFile(src, tmp)
  await chmod(tmp, 0o644)  // the source may be a read-only custody copy
  await rename(tmp, join(INBOX, fileName))
  const old = new Date(Date.now() - 5000)
  await utimes(join(INBOX, fileName), old, old)
  return createHash('sha256').update(readFileSync(src)).digest('hex')
}
const jobSha = new Map()  // job_id -> scan sha256 (the list summary carries no sha; the detail does)
async function jobForSha(sha, timeoutMs = 60000) {
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
async function jobDone(jobId, timeoutMs = 240000) {
  return waitFor(async () => {
    const r = await api('GET', `/api/line/jobs/${jobId}`)
    const job = r.json && (r.json.job || r.json)
    return job && ['completed', 'failed', 'cancelled'].includes(job.execution_status) ? job : null
  }, timeoutMs, `job ${jobId} to finish`, 250)
}

const baseEnv = {
  PATH: process.env.PATH, HOME: process.env.HOME, TMPDIR: process.env.TMPDIR || tmpdir(), LANG: 'en_US.UTF-8',
  PYTHONPATH: `${APP}:${join(APP, 'backend')}`, PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', OMP_NUM_THREADS: '1',
  SHERMAN_LINE_ENABLED: '1', SHERMAN_LINE_DATA_ROOT: DATA, SHERMAN_LINE_STATION_ID: STATION, SHERMAN_LINE_INBOX: INBOX,
  SECRET_KEY: randomBytes(32).toString('hex'), JWT_EXPIRATION_HOURS: '4',
}

await step('S0', 'build a temporary app copy (C04 --profile app) and an E2 part set', async () => {
  const b = run(PY, ['-m', 'apps.production_line.engine_bundle', 'build', '--repo-root', REPO, '--dest', APP, '--profile', 'app'],
    { cwd: REPO, env: { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend`, PYTHONDONTWRITEBYTECODE: '1' }, timeout: 900000 })
  await writeFile(join(OUT, 'logs', 'build-app-copy.log'), b.out + b.err)
  check(b.code === 0, 'app copy built (exit 0)', b.code)
  check(existsSync(join(APP, 'bundle_manifest.json')) && existsSync(join(APP, 'frontend', 'dist', 'index.html')), 'copy has a manifest and a built SPA')
  const ps = run(PY, [SUPPORT, 'partset', '--out', join(WORK, 'e2')], { cwd: REPO, env: { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend`, PYTHONDONTWRITEBYTECODE: '1', OMP_NUM_THREADS: '1' }, timeout: 600000 })
  check(ps.code === 0, 'E2 part set generated', ps.err.slice(-400))
  setDir = JSON.parse(ps.out.trim().split('\n').pop()).set_dir
  partSet = JSON.parse(readFileSync(join(setDir, 'part_set.json'), 'utf8'))
  check(partSet.scans.length >= 5, 'part set has the drop scans', partSet.scans.map(s => s.name))
})

await step('S1', 'hardening: the server refuses to start without SECRET_KEY (exit 2, SECRET_KEY_MISSING_OR_DEFAULT)', async () => {
  const env = { ...baseEnv, SHERMAN_PRODUCTION_HARDENING: '1' }
  delete env.SECRET_KEY
  const r = run(PY, ['-m', 'apps.production_line.server', '--port', String(await freePort())], { cwd: APP, env, timeout: 120000 })
  await writeFile(join(OUT, 'logs', 'server-hardening-refusal.log'), r.out + r.err)
  check(r.code === 2, 'exit code 2', r.code)
  check(/SECRET_KEY_MISSING_OR_DEFAULT/.test(r.err), 'stderr names SECRET_KEY_MISSING_OR_DEFAULT')
  check(!existsSync(join(APP, 'uploads')), 'a refused start wrote nothing under the app root (legacy never imported)')
}, { requires: ['S0'] })

await step('S2', 'bootstrap admin, start the API server and the line service from the app copy', async () => {
  const boot = spawnSync(PY, ['-m', 'apps.production_line.security.bootstrap_admin', '--users-db', join(APP, 'data', 'users.db'),
    '--username', ADMIN.username, '--password-stdin'], { cwd: APP, env: baseEnv, input: `${ADMIN.password}\n`, encoding: 'utf8' })
  check(boot.status === 0, 'bootstrap_admin exit 0', boot.stderr.slice(-300))
  const port = await freePort()
  base = `http://127.0.0.1:${port}`
  server = start('server', ['-m', 'apps.production_line.server', '--port', String(port)], baseEnv, APP)
  serviceEnv = baseEnv
  service = start('service', ['-m', 'apps.production_line.service_main', '--build-bundle'], baseEnv, APP)
  const ready = await waitFor(async () => {
    const r = await api('GET', '/api/line/health/ready', null, '')
    return r.json && r.json.ready === true ? r.json : null
  }, 240000, 'GET /api/line/health/ready -> ready')
  check(ready.ready === true && typeof ready.checks === 'object', 'readiness {ready, checks}', ready)
  check(!/token=sst1\./.test(server.logText()), 'server log has no stream token')
}, { requires: ['S0'] })

browser = await chromium.launch({ headless: !HEADED, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1, hasTouch: true, acceptDownloads: true, locale: 'en-US' })
const page = await context.newPage()
failurePages.push(['wizard', page])
const pageErrors = []
page.on('pageerror', e => pageErrors.push(String(e)))
page.setDefaultTimeout(30000)
const wizard = s => `${base}/production/parts/${encodeURIComponent(partId)}/onboard?step=${s}`
const attr = async (p, sel, name) => p.locator(sel).first().getAttribute(name)
const fileInput = p => p.locator('[data-step-body] input[type="file"], input[type="file"]').first()

await step('W1', 'admin logs in; Production -> Parts -> New part', async () => {
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
  await shot(page, 'production-parts', 'production parts list rendered ([data-production-parts])')
  await page.click('[data-new-part]')
  await page.locator('[data-new-production-part]').waitFor()
  await page.fill('[data-field="part_number"]', partSet.sku)
  await page.fill('[data-field="part_name"]', partSet.part_name)
  await page.fill('[data-field="revision"]', partSet.revision)
  await page.fill('[data-field="thickness"]', '2.5')
  check(await attr(page, '[data-sku-preview]', 'data-sku-preview') === partSet.sku, 'SKU preview equals the canonical SKU', partSet.sku)
  await shot(page, 'new-part-form', 'SKU preview attribute equals the SKU')
  await page.click('[data-create-part]')
  await page.waitForURL(/\/production\/parts\/[^/]+\/onboard\?step=cad$/)
  partId = decodeURIComponent(new URL(page.url()).pathname.split('/')[3])
  check(partId.length > 0, 'wizard opened for the new part', partId)
}, { requires: ['S2'] })

await step('W2', 'upload the folded STEP (legacy catalog route + native bend extraction)', async () => {
  await page.goto(wizard('cad'))
  await fileInput(page).setInputFiles(join(setDir, 'part.step'))
  await page.locator('[data-cad-current]').waitFor({ timeout: 300000 })
  const n = await waitFor(async () => {
    const t = (await page.locator('[data-candidate-count]').first().innerText()).trim()
    return /^\d+$/.test(t) && Number(t) > 0 ? Number(t) : null
  }, 120000, 'the CAD candidate count')
  check(n === 4, 'CAD bend candidates extracted from the STEP (HAT: 4)', n)
  await page.locator('[data-upload-phase]').waitFor({ state: 'detached', timeout: 60000 })
  const cadName = (await page.locator('[data-cad-current]').innerText()).split('\n')[0]
  note(!/^[0-9a-f]{24,}\./i.test(cadName), 'the CAD card names the uploaded file, not the stored hash name', cadName)
  await settle(page)
  await shot(page, 'wizard-cad', 'candidate count = 4; no upload card left')
}, { requires: ['W1'] })

await step('W3', 'upload the PDF drawing', async () => {
  await page.goto(wizard('drawing'))
  await fileInput(page).setInputFiles(join(setDir, 'drawing.pdf'))
  await page.locator('[data-drawing-current]').waitFor({ timeout: 60000 })
  check(await page.locator('[data-drawing-current]').count() === 1, 'drawing attached')
  await page.locator('[data-upload-phase]').waitFor({ state: 'detached', timeout: 60000 })
  await settle(page)
  check(await page.locator('[data-step-body] .onb-drop[data-busy="true"]').count() === 0, 'no stale upload card under the attached drawing')
  await shot(page, 'wizard-drawing', 'drawing current row present; upload card gone')
}, { requires: ['W1'] })

// ------------------------------------------------------------------ Bends & order through the UI
// Wizard step 5 -> "Tag bends" opens Vendor Setup embedded as the tagging studio. The bends the
// backend CAD bend solver auto-marked are drawn on the CAD; the run clicks each one ON THE MODEL
// (falling back to its list row only when the on-model marker is not clickable, recorded per
// click), confirms it, tags the order and the מהלך groups, toggles folded/unfolded, saves and
// clicks "Done, back to the wizard". Only then does it assert through the API.
const wizardFor = (id, s) => `${base}/production/parts/${encodeURIComponent(id)}/onboard?step=${s}`
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
/**
 * Drives the tagging studio for `pid`. `plan` = { order: [feature ids], moves: [[feature ids]] } or null
 * (then: the studio's own list order, one מהלך of the first two bends). Returns what it did.
 * `unfold`: 'require' (must be available), 'note' (availability is informational).
 */
async function driveStudio(p, pid, plan, prefix, { unfold = 'require' } = {}) {
  const did = { clicks: [] }
  await p.goto(wizardFor(pid, 'bends'))
  const entry = p.locator('[data-step-body] [data-tag-bends], [data-tag-bends]').first()
  await entry.waitFor({ timeout: 60000 })
  check(await p.locator('[data-wizard][data-current-step="bends"]').count() === 1, 'wizard step 5 (Bends & order) is open with a "Tag bends" action')
  await shot(p, `${prefix}-bends-entry`, 'data-current-step=bends and the [data-tag-bends] button present')
  await entry.click()
  await p.waitForURL(u => new URL(u).searchParams.get('studio') === '1', { timeout: 15000 })
  await p.locator(`${STUDIO} [data-studio-bar]`).waitFor({ timeout: 60000 })
  await p.locator(`${STUDIO} [data-bend-id]`).first().waitFor({ timeout: 120000 })
  const setup0 = (await api('GET', `/api/parts/${encodeURIComponent(pid)}/vendor-setup`)).json
  const cands = (setup0 && setup0.candidates) || []
  const rows0 = (await studioRows(p)).filter(r => r.status === 'active')
  did.candidates = cands.map(c => ({ feature_id: c.feature_id, label: c.label, target_angle: c.target_angle }))
  did.candidate_generation = setup0 && setup0.candidate_generation
  check(rows0.length > 0 && rows0.length === cands.length && rows0.every(r => cands.some(c => c.feature_id === r.id)),
    'studio lists every bend the CAD solver auto-marked (DOM rows = API candidates)', { dom: rows0.length, api: cands.length })
  check(rows0.every(r => r.review === 'pending'), 'auto-marked bends start unconfirmed (pending)', rows0.map(r => r.review))
  const markers = await waitFor(async () => {
    const n = await p.locator(`${STUDIO} [data-bend-marker]`).count()
    return n === cands.length ? n : null
  }, 60000, 'one on-model marker per auto-marked bend').catch(() => p.locator(`${STUDIO} [data-bend-marker]`).count())
  did.markers = markers
  check(markers === cands.length, 'the CAD view draws one numbered marker per auto-marked bend', { markers, candidates: cands.length })
  check(await p.locator(`${STUDIO} .vendor-setup-canvas canvas`).count() >= 1, 'the CAD viewer canvas is mounted')
  await shot(p, `${prefix}-studio-auto-marked`, `studio: ${cands.length} auto-marked bends (DOM rows = API candidates = on-model markers), all pending`)

  const labelOf = new Map(rows0.map(r => [r.id, r.label]))
  const order = plan ? plan.order : rows0.map(r => r.id)
  const moves = plan ? plan.moves : [order.slice(0, 2)]
  did.order = order.map(id => labelOf.get(id))
  did.moves = moves.map(m => m.map(id => labelOf.get(id)))
  check(order.length === rows0.length && new Set(order).size === order.length && order.every(id => labelOf.has(id)), 'the planned order names every bend once', did.order)
  check(moves.some(m => m.length === 2), 'the plan has a מהלך group of two bends', did.moves)

  // 1. Confirm each auto-marked bend: select it on the CAD, then "Confirm as step N".
  for (const id of order) {
    await clickBend(p, id, labelOf.get(id), 'Select bend', did.clicks)
    await p.locator(`${STUDIO} [data-confirm-bend="${id}"]`).click({ timeout: 15000 })
    await waitFor(async () => (await attr(p, `${STUDIO} [data-bend-id="${id}"]`, 'data-review-status')) === 'confirmed', 10000, `bend ${labelOf.get(id)} confirmed`)
  }
  const rows1 = await studioRows(p)
  check(rows1.filter(r => r.status === 'active').every(r => r.review === 'confirmed'), 'every auto-marked bend shows confirmed', rows1.map(r => `${r.label}:${r.review}`))
  check(await attr(p, `${STUDIO} [data-order-mode]`, 'data-order-confirmed') === 'true', 'confirming every bend in order set the order (data-order-confirmed=true)')
  await shot(p, `${prefix}-studio-confirmed`, 'all rows data-review-status=confirmed; order confirmed')

  // 2. Tag the order and the מהלך groups by clicking the bends first -> last on the model.
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
      await p.locator(`${STUDIO} [data-leave-without-move]`).click()  // this bend is a move of its own, ungrouped
    }
  }
  const tagged = await p.locator(`${STUDIO} [data-order-assigned]`).evaluateAll(els => els.map(e => e.getAttribute('data-order-assigned')))
  check(JSON.stringify(tagged) === JSON.stringify(order), 'order mode lists the bends in the clicked order', tagged.map(id => labelOf.get(id)))
  const move1 = (await p.locator(`${STUDIO} [data-machine-move="1"]`).first().innerText()).trim()
  check(moves[0].every(id => move1.includes(labelOf.get(id))), 'מהלך 1 names its two bends', move1)
  await shot(p, `${prefix}-studio-order-moves`, `order mode: ${tagged.length} bends tagged in order; [data-machine-move="1"] = ${move1.replace(/\s+/g, ' ')}`)
  await p.locator(`${STUDIO} [data-apply-order]`).click()
  await p.locator(`${STUDIO} [data-order-mode="false"][data-order-confirmed="true"]`).waitFor({ timeout: 10000 })
  const stepsShown = (await studioRows(p)).filter(r => r.status === 'active').sort((a, b) => Number(a.step) - Number(b.step)).map(r => r.id)
  check(JSON.stringify(stepsShown) === JSON.stringify(order), 'applied order numbers the bends 1..n in the clicked order', stepsShown.map(id => labelOf.get(id)))

  // 3. Folded / unfolded toggle.
  const unfoldBtn = p.locator(`${STUDIO} [data-view-mode="unfolded_visual"]`)
  const canUnfold = await unfoldBtn.isEnabled()
  did.unfold_available = canUnfold
  did.unfold_source = await attr(p, `${STUDIO} [data-unfold-source]`, 'data-unfold-source')
  if (unfold === 'require') check(canUnfold, 'the unfolded view is available in the studio', did.unfold_source)
  else note(canUnfold, 'the unfolded view is available in the studio (informational for this part)', did.unfold_source)
  if (canUnfold) {
    await unfoldBtn.click()
    await p.locator(`${STUDIO} [data-current-view="unfolded_visual"]`).waitFor({ timeout: 15000 })
    await sleep(800)
    check(await attr(p, `${STUDIO} [data-view-mode="unfolded_visual"]`, 'aria-pressed') === 'true', 'unfolded toggle pressed; the viewer shows the unfolded part')
    await shot(p, `${prefix}-studio-unfolded`, `data-current-view=unfolded_visual (source ${did.unfold_source})`)
    await p.locator(`${STUDIO} [data-view-mode="folded"]`).click()
    await p.locator(`${STUDIO} [data-current-view="folded"]`).waitFor({ timeout: 15000 })
    await sleep(800)
    check(await attr(p, `${STUDIO} [data-view-mode="folded"]`, 'aria-pressed') === 'true', 'folded toggle pressed; the viewer is back to the folded CAD')
    await shot(p, `${prefix}-studio-folded`, 'data-current-view=folded')
  }

  // 4. Save, then "Done, back to the wizard".
  check(await attr(p, `${STUDIO} .onb-save-state`, 'data-dirty') === 'true', 'the studio shows unsaved changes before Save')
  await p.locator(`${STUDIO} [data-save-draft]`).click()
  await waitFor(async () => (await attr(p, `${STUDIO} .onb-save-state`, 'data-dirty')) === 'false', 30000, 'the studio to show "all saved"')
  check(await p.locator(`${STUDIO} [role="alert"]`).count() === 0, 'save raised no error')
  await shot(p, `${prefix}-studio-saved`, 'save state data-dirty=false, no error alert')
  await p.locator(`${STUDIO} [data-studio-done]`).click()
  await p.waitForURL(u => new URL(u).searchParams.get('studio') !== '1', { timeout: 15000 })
  await p.locator('[data-bend-summary]').waitFor({ timeout: 60000 })

  // 5. Only now: the API and the wizard's own read of the saved draft.
  const setup = (await api('GET', `/api/parts/${encodeURIComponent(pid)}/vendor-setup`)).json
  const d = setup && setup.draft
  did.draft = d && { draft_revision: d.draft_revision, sequence: d.sequence.map(id => labelOf.get(id) || id), machine_moves: (d.machine_moves || []).map(m => m.map(id => labelOf.get(id) || id)), operator_sequence_confirmed: d.operator_sequence_confirmed, is_stale: d.is_stale }
  check(d && d.draft_revision >= 1 && !d.is_stale, 'API: a fresh Vendor Setup draft was saved', did.draft)
  check(d && d.marks.filter(m => m.status === 'active').length === cands.length && d.marks.filter(m => m.status === 'active').every(m => m.review_status === 'confirmed' && m.source === 'cad_candidate'),
    'API: every auto-marked CAD bend is saved confirmed (no manual marks)', d && d.marks.map(m => `${m.label}:${m.review_status}`))
  check(d && JSON.stringify(d.sequence) === JSON.stringify(order), 'API: saved order = the clicked order', did.draft && did.draft.sequence)
  check(d && JSON.stringify((d.machine_moves || []).filter(m => m.length > 1)) === JSON.stringify(moves.filter(m => m.length > 1)), 'API: saved מהלך groups = the tagged groups', did.draft && did.draft.machine_moves)
  check(d && d.operator_sequence_confirmed === true, 'API: operator_sequence_confirmed')
  const confirmed = await waitFor(async () => (await attr(p, '[data-sequence-confirmed]', 'data-sequence-confirmed')) === 'true', 60000, 'the wizard to show the order confirmed').catch(() => false)
  check(confirmed, 'wizard shows the order confirmed')
  const reviewed = await attr(p, '[data-marks-reviewed]', 'data-marks-reviewed')
  check(reviewed === `${cands.length}/${cands.length}`, 'wizard shows every mark reviewed', reviewed)
  const grouped = await waitFor(async () => (await p.locator('[data-move="1"] [data-bend-step]').count()) === moves[0].length, 30000, 'move 1').catch(() => false)
  check(grouped, `wizard: move 1 groups ${moves[0].length} bends`)
  await shot(p, `${prefix}-wizard-bends-order`, `wizard step 5 after Done: data-sequence-confirmed=true, ${reviewed} reviewed, move 1 has ${moves[0].length} bends`)
  return did
}

await step('W4', 'Bends & order through the UI: open Vendor Setup from step 5, confirm the auto-marked bends, order, one מהלך, folded/unfolded, back to the wizard', async () => {
  // Test oracle (read-only GET): which candidate is which E2 bend, so the clicked order is the E2 order.
  const pr = run(PY, [SUPPORT, 'vendor-plan', '--api', base, '--part-id', partId, '--set-dir', setDir],
    { cwd: REPO, env: { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend`, SHERMAN_E2E_TOKEN: token }, timeout: 120000 })
  const plan = JSON.parse(pr.out.trim().split('\n').pop() || '{}')
  current.plan = plan
  check(pr.code === 0 && plan.ok, 'the auto-marked candidates match the E2 bends one to one (test oracle)', plan)
  try {
    // Before the flat is bound the studio can only offer a CAD development, which this STEP may not have;
    // the §6 toggle (unfolded = the uploaded flat) is driven in the studio in W6b and in the wizard in W6.
    current.ui = await driveStudio(page, partId, { order: plan.order.map(o => o.feature_id), moves: plan.moves }, 'w4', { unfold: 'note' })
  } catch (err) {
    // FALLBACK ONLY: save the same draft through the API so the later steps can run. Still a FAIL.
    const r = run(PY, [SUPPORT, 'vendor-draft', '--api', base, '--part-id', partId, '--set-dir', setDir],
      { cwd: REPO, env: { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend`, SHERMAN_E2E_TOKEN: token }, timeout: 300000 })
    const doc = JSON.parse(r.out.trim().split('\n').pop() || '{}')
    current.api_fallback = doc
    current.unblocks = r.code === 0 && !!doc.ok
    throw new Error(`UI path failed (${err && err.message ? err.message.split('\n')[0] : err}); API fallback ${current.unblocks ? 'applied' : 'also failed'} - recorded FAIL`)
  }
}, { requires: ['W2', 'W3'] })

await step('W5', 'upload the flat DXF; propose and confirm the bend-line binding', async () => {
  await page.goto(wizard('flat'))
  await fileInput(page).setInputFiles(join(setDir, 'flat.dxf'))
  await page.locator('[data-flat-parse]').waitFor({ timeout: 120000 })
  check(await attr(page, '[data-flat-parse]', 'data-flat-parse') === 'parsed', 'flat parsed', await attr(page, '[data-flat-parse]', 'data-flat-parse'))
  await page.locator('[data-propose]').waitFor()
  if (await page.locator('[data-binding-chip="PROPOSED"]').count() === 0) {
    await waitFor(async () => page.locator('[data-propose]').isEnabled(), 60000, 'the propose button')
    await page.click('[data-propose]')
    await page.locator('[data-binding-chip="PROPOSED"]').waitFor({ timeout: 120000 })
  }
  current.binding_warnings = await page.locator('[data-warn]').evaluateAll(els => els.map(e => e.getAttribute('data-warn')))
  if (await page.locator('[data-binding-problem="MIRROR_UNDECIDED"]').count()) await page.check('[data-mirror="AS_PROPOSED"]').catch(() => {})
  await shot(page, 'wizard-flat-proposal', 'binding editor shown with the flat view')
  await page.click('[data-confirm-binding]')
  await page.locator('[data-binding-chip="CONFIRMED"]').waitFor({ timeout: 60000 })
  check(await page.locator('[data-binding-fresh]').count() >= 1, 'binding confirmed and fresh')
  const chipOverlaps = await page.evaluate(() => {
    const boxes = [...document.querySelectorAll('[data-flat-view] .lbl-bg')].map(r => r.getBoundingClientRect())
    let n = 0
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const [a, b] = [boxes[i], boxes[j]]
      if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1) n++
    }
    return { labels: boxes.length, overlaps: n }
  })
  check(chipOverlaps.overlaps === 0, 'flat bend-line labels do not overlap', chipOverlaps)
  await page.locator('[data-binding-chip="CONFIRMED"]').first().scrollIntoViewIfNeeded()
  await settle(page)
  await shot(page, 'wizard-flat-confirmed', 'binding chip CONFIRMED (in view); flat labels do not overlap')
}, { requires: ['W4'] })

await step('W6', 'toggle folded / unfolded (the unfolded view comes from the uploaded flat)', async () => {
  await page.goto(wizard('bends'))
  await page.click('[data-view="unfolded"]')
  await page.locator('[data-unfold-source="uploaded-flat"]').waitFor({ timeout: 30000 })
  const labels = await page.locator('[data-flat-view] text').allTextContents()
  check(labels.length === 4, 'unfolded view draws 4 numbered bend lines', labels)
  check(await attr(page, '[data-view="unfolded"]', 'aria-pressed') === 'true' && await attr(page, '[data-view="folded"]', 'aria-pressed') === 'false',
    'the Unfolded toggle is the pressed one while the unfolded view is shown')
  await settle(page)
  await shot(page, 'wizard-unfolded', 'unfold source = uploaded-flat; Unfolded aria-pressed=true')
  await page.click('[data-view="folded"]')
  await sleep(500)
  check(await page.locator('[data-unfold-source="uploaded-flat"]').count() === 0, 'folded view is back')
  check(await attr(page, '[data-view="folded"]', 'aria-pressed') === 'true', 'the Folded toggle is pressed')
  await settle(page)
  await shot(page, 'wizard-folded', 'unfolded source gone; Folded aria-pressed=true')
}, { requires: ['W5'] })

await step('W6b', 'Vendor Setup studio again: its unfolded view now comes from the uploaded flat; the studio in Hebrew (RTL)', async () => {
  await page.goto(wizard('bends'))
  await page.locator('[data-tag-bends]').first().click()
  await page.locator(`${STUDIO} [data-studio-bar]`).waitFor({ timeout: 60000 })
  await page.locator(`${STUDIO} [data-bend-id]`).first().waitFor({ timeout: 60000 })
  const src = await waitFor(async () => (await attr(page, `${STUDIO} [data-unfold-source]`, 'data-unfold-source')) === 'uploaded_flat' ? 'uploaded_flat' : null, 60000, 'the studio to use the uploaded flat').catch(async () => attr(page, `${STUDIO} [data-unfold-source]`, 'data-unfold-source'))
  check(src === 'uploaded_flat', 'Vendor Setup unfolded source = the uploaded, confirmed flat (not the CAD development)', src)
  await page.locator(`${STUDIO} [data-view-mode="unfolded_visual"]`).click()
  await page.locator(`${STUDIO} [data-current-view="unfolded_visual"]`).waitFor({ timeout: 15000 })
  await sleep(800)
  check(await attr(page, `${STUDIO} [data-view-mode="unfolded_visual"]`, 'aria-pressed') === 'true', 'studio unfolded toggle pressed')
  await shot(page, 'studio-unfolded-from-flat', 'data-unfold-source=uploaded_flat, data-current-view=unfolded_visual')
  await page.locator(`${STUDIO} [data-view-mode="folded"]`).click()
  await page.locator(`${STUDIO} [data-current-view="folded"]`).waitFor({ timeout: 15000 })
  await sleep(800)
  check(await attr(page, `${STUDIO} [data-view-mode="folded"]`, 'aria-pressed') === 'true', 'studio folded toggle pressed; folded CAD back')
  await shot(page, 'studio-folded-again', 'data-current-view=folded after the unfolded view')
  // Hebrew: the wizard's language switch is the studio's only one.
  await page.locator('[data-language="he"]').first().click()
  await page.locator(`.vendor-setup-root[data-embedded="wizard"][dir="rtl"]`).waitFor({ timeout: 15000 })
  await page.locator('[data-wizard][dir="rtl"]').waitFor()
  await sleep(800)
  const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth }))
  check(m.sw <= 1920, 'no horizontal overflow in the Hebrew studio', m)
  const he = await page.locator(`${STUDIO} [data-studio-bar]`).innerText()
  check(/[֐-׿]{3,}/.test(he), 'studio guide is in Hebrew', he.slice(0, 120))
  check(await page.locator(`${STUDIO} [data-bend-marker]`).count() >= 1, 'CAD markers still drawn in RTL')
  await shot(page, 'studio-hebrew-rtl', 'studio root dir=rtl, Hebrew guide, markers drawn, no overflow')
  await page.locator('[data-language="en"]').first().click()
  check(await attr(page, `${STUDIO} .onb-save-state`, 'data-dirty') === 'false', 'viewing and switching language left nothing unsaved')
  await page.locator(`${STUDIO} [data-studio-done]`).click()
  await page.locator('[data-bend-summary]').waitFor({ timeout: 60000 })
}, { requires: ['W5'] })

async function setTolerances(p, pid, nRows, prefix) {
  await p.goto(wizardFor(pid, 'tolerances'))
  await p.locator('[data-tolerance-editor]').waitFor()
  // The form is re-initialised when the saved release draft and the bend rows arrive.
  await waitFor(async () => (await p.locator('[data-override-row]').count()) === nRows, 60000, `the ${nRows} per-bend rows`)
  await sleep(500)
  for (const [id, v] of [['angle', '1'], ['flange', '0.5'], ['bendLine', '0.5'], ['backgauge', '0.3']]) {
    await p.fill(`#${id}-plus`, v)
    if (await p.locator(`#${id}-minus`).count()) await p.fill(`#${id}-minus`, v)
  }
  check(await p.inputValue('#angle-plus') === '1' && await p.inputValue('#flange-plus') === '0.5', 'tolerance fields filled')
  await p.selectOption('[data-tol-source]', 'SHOP_STANDARD')
  await p.check('[data-stage="FINAL"]')
  await p.click('[data-save-tolerances]')
  await p.locator('[data-tolerances-saved]').waitFor({ timeout: 30000 })
  const draft = await api('GET', `/api/production/parts/${encodeURIComponent(pid)}/release-draft`)
  const tol = draft.json && draft.json.request && draft.json.request.default_tolerances
  check(tol && tol.angle_deg.plus === 1 && tol.flange_mm.plus === 0.5, 'release draft saved with the tolerances', tol)
  await shot(p, `${prefix}-tolerances`, `release draft has angle ±1, flange ±0.5; ${nRows} per-bend rows`)
}

async function releasePart(p, pid, rid, nRows, prefix) {
  await p.goto(wizardFor(pid, 'review'))
  // Wait for the loaded wizard (not its loading state): SKU header, settled rail, review body.
  await p.locator('[data-wizard][data-loaded="true"]').waitFor({ timeout: 60000 })
  await p.locator('[data-sku-tag]').waitFor({ timeout: 60000 })
  await p.locator('[data-review-step]').waitFor()
  check(await p.locator('[data-review-step]').count() === 1, 'review step rendered')
  const rail = await attr(p, '[data-rail-progress]', 'data-rail-progress')
  check(rail !== 'loading' && !/^0\//.test(rail), 'the rail shows the real progress on the review step (no reset wizard)', rail)
  await settle(p)
  await shot(p, `${prefix}-review`, `review step rendered with the loaded wizard (rail ${rail})`)
  await p.goto(wizardFor(pid, 'release'))
  await p.locator('[data-release-button]').waitFor()
  await p.click('[data-release-button]')
  await p.locator('[data-release-result="released"], [data-release-refused]').first().waitFor({ timeout: 300000 })
  if (await p.locator('[data-release-refused]').count()) {
    const why = (await p.locator('[data-release-refused]').innerText()).slice(0, 1500)
    current.release_refusal = why
    await shot(p, `${prefix}-release-refused`, 'release refused ([data-release-refused]) - see the refusal text in steps.json')
    check(false, 'release succeeded', why)
  }
  const text = await p.locator('[data-release-result]').innerText()
  check(text.includes(rid), `release result names ${rid}`, text.slice(0, 200))
  await p.locator('[data-expectations-status="READY"]').first().waitFor({ timeout: 60000 })
  const rows = await p.locator('[data-expectation-row]').count()
  check(rows === nRows, `expectations table has ${nRows} bend rows`, rows)
  await p.locator('[data-release-button][data-release-state="released"]').waitFor({ timeout: 30000 })
  check(!/Releasing/.test(await p.locator('.onb-release-card').innerText()), 'no lingering "Releasing…" once released')
  await settle(p)
  await shot(p, `${prefix}-released`, `release result names ${rid}; button shows Released`)
  await p.locator('[data-expectation-row]').first().scrollIntoViewIfNeeded()
  await sleep(300)
  await shot(p, `${prefix}-released-expectations`, `expectations READY with ${nRows} rows (nominal flange lengths and angles)`)
  current.expectations = await p.locator('[data-expectation-row]').evaluateAll(els => els.map(e => e.innerText.replace(/\s+/g, ' ').trim()).slice(0, 40))
  const rec = await api('GET', `/api/production/recipes/${encodeURIComponent(rid)}`)
  check(rec.status === 200 && rec.json.benchmark_reference && rec.json.benchmark_reference.status === 'REGISTERED',
    'recipe API reports a REGISTERED benchmark reference', rec.json && rec.json.benchmark_reference)
  const bm = run(PY, [SUPPORT, 'benchmark', '--data-root', DATA, '--recipe-id', rid], { cwd: REPO, env: { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend` } })
  const bdoc = JSON.parse(bm.out.trim().split('\n').pop() || '{}')
  current.benchmark = bdoc
  check(bm.code === 0 && bdoc.ok, 'benchmark_references row has per-bend nominals for every bend', bdoc)
  return rec.json
}

await step('W7', 'set tolerances (FINAL inspection)', async () => {
  await setTolerances(page, partId, 4, 'wizard')
}, { requires: ['W4'] })

await step('W8', 'review, release SKU@v1; expectations show nominal flange lengths and angles; benchmark row', async () => {
  recipeId = `${partSet.sku}@v1`
  await releasePart(page, partId, recipeId, 4, 'wizard')
}, { requires: ['W7'] })

await step('W9', 'wizard in Hebrew (RTL)', async () => {
  await page.goto(wizard('release'))
  await page.locator('[data-language="he"]').first().click()
  await page.locator('[data-wizard][dir="rtl"]').waitFor()
  // The page was reloaded: wait for the loaded snapshot (part header, the active recipe and the
  // completed checklist), never screenshot the transient pre-load state.
  await page.locator('[data-sku-tag]').waitFor({ timeout: 60000 })
  await page.locator(`[data-active-recipe="${recipeId}"], [data-release-unchanged]`).first().waitFor({ timeout: 60000 })
  const doneKeys = ['part', 'cad', 'drawing', 'flat', 'bends', 'tolerances']
  const loaded = await waitFor(async () => {
    const st = await page.locator('[data-checklist] [data-check]').evaluateAll(els => Object.fromEntries(els.map(e => [e.getAttribute('data-check'), e.getAttribute('data-status')])))
    return doneKeys.every(k => st[k] === 'done') ? st : null
  }, 60000, 'the Hebrew checklist to show the completed steps').catch(() => null)
  current.checklist = loaded
  check(!!loaded, 'Hebrew checklist shows part/cad/drawing/flat/bends/tolerances done', loaded)
  const sku = (await page.locator('[data-sku-tag]').innerText()).trim()
  check(sku.includes(partSet.sku), 'part header names the SKU', sku)
  const falseBlockers = await page.locator('[data-reason-code="CAD_MISSING"], [data-reason-code="DRAWING_MISSING"], [data-reason-code="TOLERANCES_MISSING"]').count()
  check(falseBlockers === 0, 'no CAD/drawing/tolerances "missing" blockers on the released part', falseBlockers)
  const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth }))
  check(m.sw <= 1920, 'no horizontal overflow in RTL', m)
  await shot(page, 'wizard-hebrew-rtl', 'wizard root dir=rtl; loaded: SKU header, active recipe, checklist done, no missing blockers')
  await page.locator('[data-language="en"]').first().click()
}, { requires: ['W8'] })

// ------------------------------------------------------------------ station
// The kiosk is a separate browser profile: it must not see the admin's stored user token.
const kioskContext = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1, hasTouch: true, acceptDownloads: true, locale: 'en-US' })
const kiosk = await kioskContext.newPage()
failurePages.push(['station', kiosk])
kiosk.on('pageerror', e => pageErrors.push(`kiosk: ${String(e)}`))
kiosk.setDefaultTimeout(30000)
const sattr = (sel, name) => attr(kiosk, sel, name)
let deviceToken = ''
let jobErrors = null

await step('K1', 'station kiosk: device-token gate, then the connected empty station', async () => {
  await kiosk.goto(`${base}/station/${STATION}`)
  await kiosk.locator('[data-testid="station-auth-gate"]').waitFor()
  await shot(kiosk, 'station-token-gate', 'auth gate shown without a stored token')
  const r = await api('POST', '/api/line/devices', { station_id: STATION, label: 'e2e-kiosk' })
  check(r.status === 201 || r.status === 200, 'device token issued by an admin', r.status)
  deviceToken = r.json.token
  await kiosk.fill('#st-token', deviceToken)
  await kiosk.press('#st-token', 'Enter')
  await kiosk.locator('[data-testid="station-root"]').waitFor()
  check(await kiosk.locator('[data-testid="station-root"]').count() === 1, 'station connected with the device token')
  // Evidence of "connected": wait for the live stream (or at least a first polled feed), never "Connecting…".
  const conn = await waitFor(async () => (await sattr('[data-testid="station-root"]', 'data-connection')) === 'live' ? 'live' : null, 20000, 'the live connection')
    .catch(async () => sattr('[data-testid="station-root"]', 'data-connection'))
  note(conn === 'live', 'station connection is Live (stream open)', conn)
  await waitFor(async () => !/Connecting/.test(await kiosk.locator('[data-testid="station-root"]').innerText()), 20000, 'the first feed')
  await settle(kiosk)
  await shot(kiosk, 'station-connected', `station root rendered, connection ${conn}, feed loaded`)
}, { requires: ['S2'] })

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

await step('K2', 'ZEISS-style drop of the error scan (+1.5 mm B2 flange, +2° B3); station updates within 5 s', async () => {
  const e = scan('errors')
  const sha = await dropScan('errors', e.zeiss_filename)
  // The job lookup (API) runs concurrently with the DOM poll below, so the in-progress state is
  // watched from the moment of the drop.
  const jobLookup = jobForSha(sha)
  jobLookup.catch(() => {})  // awaited below; avoid an unhandled rejection while the DOM poll runs
  // Amendment 10: the in-progress state must be seen and captured. The kiosk is already
  // connected and watching; poll the DOM (not the API) for the active ribbon while the job runs.
  const inProgress = await waitFor(async () => {
    if (await kiosk.locator('[data-testid="status-ribbon"][data-ribbon="active"]').count()) {
      return { ribbon: 'active', stage: await sattr('[data-testid="status-ribbon"]', 'data-job-stage'), elapsed: await kiosk.locator('[data-testid="elapsed"]').count() }
    }
    return null
  }, 60000, 'the active-job ribbon on the station', 100).catch(() => null)
  current.in_progress = inProgress
  if (inProgress) await shot(kiosk, 'station-in-progress', 'status ribbon data-ribbon=active with the job stage and elapsed timer')
  check(!!inProgress && inProgress.elapsed === 1, 'station showed the in-progress state (active ribbon + elapsed timer) while the job ran', inProgress)
  const job = await jobLookup
  const done = await jobDone(job.job_id)
  check(done.execution_status === 'completed', 'job completed', done.failure || done.execution_status)
  const finished = Date.parse(done.finished_at)
  await kiosk.locator(`[data-testid="station-job-id"][data-job-id="${job.job_id}"]`).waitFor({ timeout: 15000 })
  const lag = (Date.now() - finished) / 1000
  current.update_lag_s = lag
  check(lag <= 5.0, 'station shows the new job within 5 s of the job finishing, no reload', lag)
  jobErrors = job.job_id
  await waitFor(async () => (await sattr('[data-testid="station-viewer"]', 'data-scan-colour')) === 'heatmap', 20000, 'heatmap colours')
  await sleep(800)
  const px = await canvasPixels(kiosk)
  // The part fills only part of the viewer; "not blank" = a few hundred painted probe pixels in many colours.
  check(px.painted > 300 && px.colours > 20, 'WebGL canvas is not blank', px)
  const state = await sattr('[data-testid="summary-banner"]', 'data-summary-state')
  check(['OUT_OF_TOLERANCE_FOUND', 'REVIEW_REQUIRED'].includes(state), 'verdict banner shows the errors', state)
  const res = await api('GET', `/api/line/jobs/${job.job_id}/result`)
  const result = res.json
  current.engine_summary = result.summary
  const injected = e.expected.injected
  const neverPass = new Set(e.expected.characteristics_never_pass)
  const rows = []
  for (const b of result.bends) for (const c of b.characteristics) {
    const key = `${c.bend_uid}/${c.kind}${c.side ? `/${c.side}` : ''}`
    if (neverPass.has(key)) rows.push({ key, label: b.label, seq: b.sequence_index, decision: c.decision, state: c.state, actual: c.actual, truth: injected[key] && injected[key].truth })
  }
  current.injected_rows = rows
  check(rows.length === 2, 'both injected characteristics present', rows)
  for (const r of rows) {
    check(['FAIL', 'REVIEW'].includes(r.decision), `${r.label} ${r.key.split('/').slice(1).join('/')} is FAIL or REVIEW`, r.decision)
    if (r.state === 'MEASURED') check(Math.abs(r.actual - r.truth) <= (r.key.includes('ANGLE') ? 0.3 : 0.2), `${r.label} actual within tolerance of truth`, r)
    const cell = r.key.includes('ANGLE') ? `cell-${r.seq}-angle` : `cell-${r.seq}-flange-${r.key.endsWith('/A') ? 'a' : 'b'}`
    check(['FAIL', 'REVIEW'].includes(await sattr(`[data-testid="${cell}"]`, 'data-decision')), `table cell ${cell} shows the decision`, await sattr(`[data-testid="${cell}"]`, 'data-decision'))
  }
  const lp = await kiosk.evaluate(probeLabels)
  const lprob = labelProblems(lp, { minVisible: 4 })
  check(lprob.length === 0, 'bend labels: no overlap, no mis-tag, none over the viewer controls', lprob.length ? lprob : lp && { visible: lp.visible })
  await settle(kiosk)
  await shot(kiosk, 'station-result-errors', 'banner OUT_OF_TOLERANCE_FOUND/REVIEW; injected cells FAIL/REVIEW; canvas painted; labels clean')
}, { requires: ['K1', 'W8'] })

await step('K3', 'station layers: heatmap off/on, CAD only, scan only, both, labels off', async () => {
  await kiosk.click('[data-layer="heatmap"]')
  await waitFor(async () => (await sattr('[data-testid="station-viewer"]', 'data-scan-colour')) === 'neutral', 5000, 'neutral scan colour')
  check(await sattr('[data-testid="heatmap-legend"]', 'data-heatmap') === 'off', 'legend says heatmap off')
  await shot(kiosk, 'station-heatmap-off', 'data-scan-colour=neutral, legend off')
  await kiosk.click('[data-layer="heatmap"]')
  await waitFor(async () => (await sattr('[data-testid="station-viewer"]', 'data-scan-colour')) === 'heatmap', 5000, 'heatmap again')
  await shot(kiosk, 'station-heatmap-on', 'data-scan-colour=heatmap')
  await kiosk.click('[data-layer="scan"]')
  await waitFor(async () => (await sattr('[data-testid="station-viewer"]', 'data-scan-colour')) === 'none', 5000, 'scan hidden')
  const cadPx = await canvasPixels(kiosk)
  check(cadPx.painted > 300 && cadPx.colours > 5, 'CAD-only view is not blank', cadPx)
  await shot(kiosk, 'station-cad-only', 'scan layer off (data-scan-colour=none), canvas painted')
  await kiosk.click('[data-layer="scan"]')
  await kiosk.click('[data-layer="cad"]')
  await waitFor(async () => (await sattr('[data-layer="cad"]', 'aria-pressed')) === 'false'
    && (await sattr('[data-testid="station-viewer"]', 'data-scan-colour')) === 'heatmap', 5000, 'scan only')
  await sleep(400)
  const scanPx = await canvasPixels(kiosk)
  check(await sattr('[data-layer="cad"]', 'aria-pressed') === 'false' && await sattr('[data-layer="scan"]', 'aria-pressed') === 'true', 'scan-only: CAD layer off, scan layer on')
  check(await sattr('[data-testid="station-viewer"]', 'data-scan-colour') !== 'none' && scanPx.painted > 300, 'scan-only view draws the scan', scanPx)
  await shot(kiosk, 'station-scan-only', 'CAD aria-pressed=false, scan aria-pressed=true, data-scan-colour=heatmap, canvas painted')
  await kiosk.click('[data-layer="cad"]')
  await waitFor(async () => (await sattr('[data-layer="cad"]', 'aria-pressed')) === 'true', 5000, 'both layers')
  await sleep(400)
  const bothPx = await canvasPixels(kiosk)
  check(await sattr('[data-layer="cad"]', 'aria-pressed') === 'true' && await sattr('[data-layer="scan"]', 'aria-pressed') === 'true'
    && await sattr('[data-testid="station-viewer"]', 'data-scan-colour') === 'heatmap', 'both layers on')
  check(bothPx.painted > 300, 'CAD + scan view is not blank', bothPx)
  await shot(kiosk, 'station-cad-and-scan', 'CAD and scan aria-pressed=true, data-scan-colour=heatmap')
  const before = await kiosk.locator('.st-label').count()
  await kiosk.click('[data-layer="labels"]')
  await waitFor(async () => (await kiosk.locator('.st-label').count()) === 0, 5000, 'labels hidden')
  check(before === 4, '4 bend labels before toggling', before)
  await shot(kiosk, 'station-labels-off', 'no .st-label elements')
  await kiosk.click('[data-layer="labels"]')
}, { requires: ['K2'] })

await step('K4', 'click bend row 2 (camera focuses), worst-first jump, orbit/zoom/touch before/after', async () => {
  const t0 = await sattr('[data-testid="station-viewer"]', 'data-camera-target')
  await kiosk.click('tr[data-sequence="2"]')
  await sleep(900)
  const t1 = await sattr('[data-testid="station-viewer"]', 'data-camera-target')
  check(t1 !== t0, 'camera target moved to bend 2', { t0, t1 })
  await shot(kiosk, 'station-row2-focus', 'camera target changed; bend 2 selected')
  // Worst-first must visibly change the selection: start from bend 1 (not the worst one).
  await kiosk.click('tr[data-sequence="1"]')
  await sleep(900)
  const before = await sattr('[data-testid="station-viewer"]', 'data-selected-bend')
  await kiosk.click('[data-action="worst"]')
  await sleep(900)
  const worst = await sattr('[data-testid="station-viewer"]', 'data-selected-bend')
  check(!!worst, 'worst-first selected a bend', worst)
  check(worst !== before, 'worst-first moved the selection away from bend 1', { before, worst })
  const wp = labelProblems(await kiosk.evaluate(probeLabels), { selectedUid: worst })
  check(wp.length === 0, 'the highlighted label is the worst bend, on top, nothing mis-tagged', wp)
  await settle(kiosk)
  await shot(kiosk, 'station-worst-first', `selected bend ${worst} (was ${before})`)
  await kiosk.click('[data-action="fit"]')
  await sleep(800)
  const box = await kiosk.locator('[data-testid="station-viewer"]').boundingBox()
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2
  const d0 = Number(await sattr('[data-testid="station-viewer"]', 'data-camera-distance'))
  await shot(kiosk, 'station-before-orbit', 'camera at fit')
  for (let i = 0; i < 6; i++) { await kiosk.mouse.move(cx, cy); await kiosk.mouse.wheel(0, -120); await sleep(60) }
  await sleep(600)
  const d1 = Number(await sattr('[data-testid="station-viewer"]', 'data-camera-distance'))
  check(d1 < d0 * 0.95, 'wheel zooms in', { d0, d1 })
  await shot(kiosk, 'station-after-zoom', 'camera distance decreased')
  const cdp = await kioskContext.newCDPSession(kiosk)
  const touch = (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts.map(([x, y], id) => ({ x, y, id })) })
  await touch('touchStart', [[cx - 40, cy], [cx + 40, cy]])
  for (let i = 1; i <= 8; i++) { await touch('touchMove', [[cx - 40 - i * 25, cy], [cx + 40 + i * 25, cy]]); await sleep(30) }
  await touch('touchEnd', [])
  await sleep(700)
  const d2 = Number(await sattr('[data-testid="station-viewer"]', 'data-camera-distance'))
  check(d2 < d1 * 0.95, 'two-finger pinch zooms in', { d1, d2 })
  const pt0 = await sattr('[data-testid="station-viewer"]', 'data-camera-position')
  await touch('touchStart', [[cx, cy]])
  for (let i = 1; i <= 8; i++) { await touch('touchMove', [[cx + i * 20, cy + i * 6]]); await sleep(30) }
  await touch('touchEnd', [])
  await sleep(600)
  const pt1 = await sattr('[data-testid="station-viewer"]', 'data-camera-position')
  check(pt1 !== pt0, 'one-finger touch drag orbits the camera', { pt0, pt1 })
  await shot(kiosk, 'station-after-touch', 'pinch reduced the distance; one-finger drag rotated')
  await kiosk.mouse.move(cx, cy); await kiosk.mouse.down(); await kiosk.mouse.move(cx + 220, cy + 60, { steps: 12 }); await kiosk.mouse.up()
  await sleep(600)
  const p1 = await sattr('[data-testid="station-viewer"]', 'data-camera-position')
  check(p1 !== pt1, 'mouse drag orbits the camera', { pt1, p1 })
  await shot(kiosk, 'station-after-orbit', 'camera position changed by the mouse drag')
  await kiosk.click('[data-action="fit"]')
  await sleep(600)
}, { requires: ['K2'] })

await step('K5', 'per-bend close-ups in the station 3D view (G4 LANDS / TAG_OK review material)', async () => {
  for (let seq = 1; seq <= 4; seq++) {
    await kiosk.click(`tr[data-sequence="${seq}"]`)
    await sleep(900)
    for (let i = 0; i < 4; i++) { await kiosk.mouse.wheel(0, -120); await sleep(50) }
    await sleep(500)
    const sel = await sattr('[data-testid="station-viewer"]', 'data-selected-bend')
    check(!!sel, `bend ${seq} selected`, sel)
    const cp = labelProblems(await kiosk.evaluate(probeLabels), { selectedUid: sel, minVisible: 1 })
    check(cp.length === 0, `close-up B${seq}: the highlighted label is B${seq}'s, nothing mis-tagged or over the controls`, cp)
    await shot(kiosk, `station-closeup-B${seq}`, `bend ${seq} (${sel}) focused; reviewer judges LANDS / TAG_OK`)
    await kiosk.click('[data-action="fit"]')
    await sleep(500)
  }
}, { requires: ['K2'] })

await step('K6', 'export PDF from the station; render pages; Hebrew and English text', async () => {
  const [download] = await Promise.all([kiosk.waitForEvent('download', { timeout: 30000 }), kiosk.click('[data-testid="download-pdf"]')])
  const pdfPath = join(OUT, 'pdf', 'report.pdf')
  await download.saveAs(pdfPath)
  const r = run(PY, [SUPPORT, 'pdf-pages', '--pdf', pdfPath, '--out', join(OUT, 'pdf'), '--expect', partSet.sku, '--expect', 'לא הוכרע', '--header-ids', partSet.sku, recipeId],
    { cwd: REPO, env: { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend` } })
  const doc = JSON.parse(r.out.trim().split('\n').pop() || '{}')
  current.pdf = doc
  check(doc.pages && doc.pages.length >= 1, 'PDF rendered to page images', doc.pages && doc.pages.length)
  check(doc.expect_found && doc.expect_found[partSet.sku], 'PDF text contains the SKU')
  check(doc.expect_found && doc.expect_found['לא הוכרע'] && doc.expect_reversed_only && doc.expect_reversed_only['לא הוכרע'] === false,
    'PDF Hebrew label "לא הוכרע" appears in logical order (not only reversed)', { found: doc.expect_found, reversed_only: doc.expect_reversed_only })
  check(doc.header && doc.header.pages_wrong.length === 0, 'PDF running header reads "<Hebrew title> · SKU · recipe id" on every page', doc.header)
  check(r.code === 0 && doc.ok, 'pdf-pages helper verdict ok', { code: r.code, ok: doc.ok })
  const he = /[֐-׿]{3,}/.test(readFileSync(join(OUT, 'pdf', 'pdf_text.txt'), 'utf8'))
  check(he, 'PDF contains Hebrew text (vision reviewer also checks glyph order)')
}, { requires: ['K2'] })

await step('K7', 'station in Hebrew (RTL)', async () => {
  await kiosk.click('[data-testid="lang-toggle"]')
  await kiosk.locator('[data-testid="station-root"][dir="rtl"]').waitFor()
  const m = await kiosk.evaluate(() => ({ sw: document.documentElement.scrollWidth, sh: document.documentElement.scrollHeight }))
  check(m.sw <= 1920 && m.sh <= 1080, 'RTL station has no page scroll', m)
  await shot(kiosk, 'station-hebrew-rtl', 'station-root dir=rtl, no overflow')
  await kiosk.click('[data-testid="lang-toggle"]')
  await kiosk.locator('[data-testid="station-root"][dir="ltr"]').waitFor()
}, { requires: ['K2'] })

await step('K8', 'incomplete-scan drop: NOT_YET_FORMED bends greyed, stage decision, backgauge lengths', async () => {
  const e = scan('stage_k')
  const sha = await dropScan('stage_k', e.zeiss_filename)
  const job = await jobForSha(sha)
  const done = await jobDone(job.job_id)
  check(done.execution_status === 'completed', 'job completed', done.failure || done.execution_status)
  await kiosk.locator(`[data-testid="station-job-id"][data-job-id="${job.job_id}"]`).waitFor({ timeout: 15000 })
  const res = (await api('GET', `/api/line/jobs/${job.job_id}/result`)).json
  current.engine_summary = { validity: res.inspection_validity.state, stage: res.stage_decision.status, summary: res.summary.state }
  const nyf = res.bends.filter(b => b.state === 'NOT_YET_FORMED')
  const truth = e.expected.bend_states_contract
  check(res.inspection_validity.state === 'VALID', 'inspection valid', res.inspection_validity)
  check(nyf.length >= 1 && nyf.every(b => truth[b.bend_uid] === 'NOT_YET_FORMED'), 'engine NOT_YET_FORMED bends match truth', nyf.map(b => b.label))
  for (const b of nyf) check(await kiosk.locator(`[data-testid="cell-${b.sequence_index}-unformed"]`).count() === 1, `B${b.sequence_index} greyed as not yet formed in the table`)
  check(!['ALL_MEASURED_IN_TOLERANCE', 'STAGE_IN_TOLERANCE'].includes(res.summary.state), 'incomplete part is never green', res.summary.state)
  check(await kiosk.locator('[data-testid="stage-strip"]').count() === 1, 'stage decision strip shown')
  const bg = res.bends.flatMap(b => b.characteristics.filter(c => c.kind === 'BACKGAUGE_LENGTH').map(c => ({ label: b.label, state: c.state, decision: c.decision, actual: c.actual, expected: c.expected })))
  current.backgauge = bg
  check(bg.length > 0, 'the engine reports backgauge lengths for the incomplete scan', bg)
  const shownBg = res.bends.filter(b => b.characteristics.some(c => c.kind === 'BACKGAUGE_LENGTH' && c.state !== 'NOT_APPLICABLE')).length
  const domBg = await kiosk.locator('[data-testid="station-panel"] .st-detail[data-kind="BACKGAUGE_LENGTH"]').count()
  check(shownBg >= 1 && domBg === shownBg, 'the table shows a backgauge row for every applicable backgauge length', { shownBg, domBg })
  await settle(kiosk)
  await shot(kiosk, 'station-incomplete', 'NOT_YET_FORMED cells greyed; stage strip; backgauge detail row; summary not green')
}, { requires: ['K1', 'W8'] })

await step('K9', 'wrong-SKU drop ends in Quarantine with a reason; the admin assigns the SKU; the job runs', async () => {
  const e = scan('unknown_sku')
  const sha = await dropScan('unknown_sku', e.zeiss_filename)
  const item = await waitFor(async () => {
    const r = await api('GET', '/api/line/quarantine')
    return ((r.json && r.json.items) || []).find(i => i.scan_file.sha256 === sha) || null
  }, 60000, 'the quarantined scan')
  check(item.quarantine_reason === 'UNMATCHED_SKU', 'quarantine reason UNMATCHED_SKU', item.quarantine_reason)
  const ribbon = await waitFor(async () => {
    const el = kiosk.locator('[data-testid="status-ribbon"][data-ribbon="issue"]')
    return (await el.count()) ? { reason: await el.getAttribute('data-reason'), text: await el.innerText() } : null
  }, 15000, 'the quarantine issue ribbon').catch(() => null)
  current.ribbon = ribbon
  check(!!ribbon && ribbon.reason === 'UNMATCHED_SKU', 'status ribbon carries the quarantine reason UNMATCHED_SKU', ribbon)
  check(!!ribbon && ribbon.text.includes(e.zeiss_filename), 'status ribbon names the dropped file', ribbon && ribbon.text)
  check(!!ribbon && /held back/i.test(ribbon.text) && /not released/i.test(ribbon.text), 'status ribbon shows the quarantine reason copy', ribbon && ribbon.text)
  await shot(kiosk, 'station-quarantine-issue', 'ribbon data-ribbon=issue, data-reason=UNMATCHED_SKU, names the file')
  const r = await api('POST', `/api/line/quarantine/${item.scan_file.scan_file_id}/assign`, { sku: partSet.sku })
  check(r.status === 200 && r.json.job_id, 'assigned by SKU -> inspect job', r.json)
  const done = await jobDone(r.json.job_id)
  check(done.execution_status === 'completed', 'assigned job completed', done.failure || done.execution_status)
  await kiosk.locator(`[data-testid="station-job-id"][data-job-id="${r.json.job_id}"]`).waitFor({ timeout: 15000 })
  await shot(kiosk, 'station-after-assign', 'station shows the assigned job')
}, { requires: ['K1', 'W8'] })

await step('K10', 'kill the line service mid-job, restart it: the job recovers and completes', async () => {
  const e = scan('stage_0')
  const sha = await dropScan('stage_0', e.zeiss_filename)
  const job = await jobForSha(sha)
  await waitFor(async () => {
    const j = (await api('GET', `/api/line/jobs/${job.job_id}`)).json
    const jj = j && (j.job || j)
    return jj && jj.execution_status === 'running' && ['engine', 'postprocess', 'preparing'].includes(jj.stage)
  }, 60000, 'the job to be running', 100)
  service.kill('SIGKILL')
  await new Promise(r => service.exitCode !== null || service.signalCode ? r() : service.once('exit', r))
  check(service.signalCode === 'SIGKILL', 'line service killed (SIGKILL) while the job was running', service.signalCode)
  const mid = (await api('GET', `/api/line/jobs/${job.job_id}`)).json
  current.state_after_kill = (mid.job || mid).execution_status
  check(current.state_after_kill === 'running', 'the killed service left the job running (lease not released)', current.state_after_kill)
  // The orphaned job must be visible on the station as the in-flight job (never "Waiting for the next scan").
  const orphan = await waitFor(async () => (await kiosk.locator('[data-testid="status-ribbon"][data-ribbon="active"]').count()) ? true : null, 15000, 'the in-flight job ribbon').catch(() => false)
  check(orphan, 'the station ribbon shows the orphaned job as in flight', await sattr('[data-testid="status-ribbon"]', 'data-ribbon'))
  await shot(kiosk, 'station-service-killed', 'job execution_status=running after the service was killed; ribbon shows the in-flight job')
  service = start('service-restart', ['-m', 'apps.production_line.service_main'], serviceEnv, APP)
  const done = await jobDone(job.job_id, 300000)
  check(done.execution_status === 'completed', 'recovered job completed', done.failure || done.execution_status)
  check(done.attempts === 2, 'second attempt after recovery', done.attempts)
  const ev = (await api('GET', `/api/line/events?after_id=0&limit=1000`)).json
  const kinds = ((ev && ev.events) || []).filter(x => x.job_id === job.job_id).map(x => x.kind)
  check(kinds.includes('JOB_RECOVERED'), 'JOB_RECOVERED event recorded', kinds)
  const res = (await api('GET', `/api/line/jobs/${job.job_id}/result`)).json
  current.engine_summary = res && { validity: res.inspection_validity.state, stage: res.stage_decision.status, inferred_stage: res.stage_decision.inferred_stage, summary: res.summary.state }
  await kiosk.locator(`[data-testid="station-job-id"][data-job-id="${job.job_id}"]`).waitFor({ timeout: 15000 })
  await sleep(3500)  // the new-arrival highlight lasts 3 s
  await settle(kiosk)
  await shot(kiosk, 'station-recovered', 'station shows the recovered job (blank, stage 0)')
}, { requires: ['K1', 'W8'] })

await step('K11', 'a duplicate drop is idempotent (SCAN_DUPLICATE, no new job)', async () => {
  const before = (await api('GET', `/api/line/jobs?station_id=${STATION}&limit=200`)).json
  const nBefore = ((before && before.jobs) || []).length
  const e = scan('errors')
  await dropScan('errors', e.zeiss_filename.replace(' 1.ply', ' 2.ply'))
  const dup = await waitFor(async () => {
    const ev = (await api('GET', `/api/line/events?after_id=0&limit=1000`)).json
    return ((ev && ev.events) || []).find(x => x.kind === 'SCAN_DUPLICATE') || null
  }, 60000, 'a SCAN_DUPLICATE event')
  check(!!dup, 'SCAN_DUPLICATE recorded')
  await sleep(2000)
  const after = (await api('GET', `/api/line/jobs?station_id=${STATION}&limit=200`)).json
  const nAfter = ((after && after.jobs) || []).length
  check(nAfter === nBefore, 'no new job for the duplicate', { nBefore, nAfter })
  const left = (await readdir(INBOX)).filter(n => !n.startsWith('.'))
  check(left.length === 0, 'inbox emptied', left)
}, { requires: ['K2'] })

// ------------------------------------------------------------------ real dev part 10839000
// Authorized Gold11 development part (plan §9; engineering-ledger evidence only). Read-only copies
// of its STEP and FULL scan go into the temp directory; nothing is written next to the originals.
const realDir = join(WORK, 'real')
let realId = ''
let realRecipe = ''
let realBends = 0
let realJob = ''
let realSku = ''

await step('R1', 'real 10839000: new part, upload its STEP (auto-marked bends) and a placeholder drawing', async () => {
  check(existsSync(REAL.step) && existsSync(REAL.fullScan), 'authorized 10839000 STEP and FULL scan present in ../source-assets/02_CAD_QC_3D')
  await mkdir(realDir, { recursive: true })
  const stp = join(realDir, `${REAL.stem}.stp`)
  const ply = join(realDir, `${REAL.stem}_full_scan.ply`)
  await copyFile(REAL.step, stp); await chmod(stp, 0o444)
  await copyFile(REAL.fullScan, ply); await chmod(ply, 0o444)
  const sha = f => createHash('sha256').update(readFileSync(f)).digest('hex')
  current.custody = { step_sha256: sha(stp), full_scan_sha256: sha(ply), source_untouched: sha(REAL.step) === sha(stp) && sha(REAL.fullScan) === sha(ply) }
  check(current.custody.source_untouched, 'read-only temp copies are byte-identical to the sources', current.custody)
  const pdf = join(realDir, `${REAL.stem}_placeholder_drawing.pdf`)
  const ph = run(PY, [SUPPORT, 'placeholder-drawing', '--out', pdf, '--title', `${REAL.stem} - no drawing on file`], { cwd: REPO, env: { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend` } })
  check(ph.code === 0 && existsSync(pdf), 'placeholder drawing PDF written', ph.err.slice(-300))

  await page.goto(`${base}/production/parts`)
  await page.locator('[data-production-parts]').waitFor()
  await page.click('[data-new-part]')
  await page.locator('[data-new-production-part]').waitFor()
  await page.fill('[data-field="part_number"]', REAL.part_number)
  await page.fill('[data-field="part_name"]', REAL.part_name)
  await page.fill('[data-field="revision"]', REAL.revision)
  realSku = await attr(page, '[data-sku-preview]', 'data-sku-preview')
  check(realSku === REAL.part_number, 'SKU preview = 10839000', realSku)
  await shot(page, 'real-new-part-form', 'SKU preview attribute = 10839000')
  await page.click('[data-create-part]')
  await page.waitForURL(/\/production\/parts\/[^/]+\/onboard\?step=cad$/)
  realId = decodeURIComponent(new URL(page.url()).pathname.split('/')[3])
  await fileInput(page).setInputFiles(stp)
  await page.locator('[data-cad-current]').waitFor({ timeout: 600000 })
  const n = await waitFor(async () => {
    const t = (await page.locator('[data-candidate-count]').first().innerText()).trim()
    return /^\d+$/.test(t) && Number(t) > 0 ? Number(t) : null
  }, 300000, 'the CAD candidate count')
  const vs = (await api('GET', `/api/parts/${encodeURIComponent(realId)}/vendor-setup`)).json
  current.cad = { candidates: n, candidate_generation: vs && vs.candidate_generation, labels: ((vs && vs.candidates) || []).map(c => `${c.label}:${c.target_angle}`) }
  check(n > 0 && vs && vs.candidates.length === n, 'the CAD bend solver auto-marked bends on the real STEP (wizard count = API candidates)', current.cad)
  await shot(page, 'real-wizard-cad', `candidate count = ${n} (= API candidates)`)
  realBends = n
  await page.goto(wizardFor(realId, 'drawing'))
  await fileInput(page).setInputFiles(pdf)
  await page.locator('[data-drawing-current]').waitFor({ timeout: 60000 })
  check(await page.locator('[data-drawing-current]').count() === 1, 'placeholder drawing attached')
}, { requires: ['S2', 'W1'] })

await step('R2', 'real 10839000: Bends & order through the Vendor Setup studio (confirm all auto-marked bends, order, one מהלך)', async () => {
  current.ui = await driveStudio(page, realId, null, 'real', { unfold: 'note' })
  realBends = current.ui.candidates.length
}, { requires: ['R1'] })

await step('R3', 'real 10839000: tolerances, review, release 10839000@v1, benchmark row', async () => {
  await setTolerances(page, realId, realBends, 'real')
  realRecipe = `${realSku}@v1`
  const rec = await releasePart(page, realId, realRecipe, realBends, 'real')
  current.recipe_bends = rec && rec.recipe && rec.recipe.bends ? rec.recipe.bends.length : null
}, { requires: ['R2'] })

await step('R4', 'real 10839000: drop its FULL scan (ZEISS-style name); the job completes and the station shows it', async () => {
  const sha = await dropFile(join(realDir, `${REAL.stem}_full_scan.ply`), REAL.zeiss)
  const job = await jobForSha(sha, 120000)
  realJob = job.job_id
  const t0 = Date.now()
  const done = await jobDone(job.job_id, 900000)
  current.job = { job_id: job.job_id, execution_status: done.execution_status, attempts: done.attempts, wall_s: Math.round((Date.now() - t0) / 1000), failure: done.failure || null }
  check(done.execution_status === 'completed', 'real FULL-scan job completed', current.job)
  const finished = Date.parse(done.finished_at)
  await kiosk.locator(`[data-testid="station-job-id"][data-job-id="${job.job_id}"]`).waitFor({ timeout: 15000 })
  current.update_lag_s = (Date.now() - finished) / 1000
  check(current.update_lag_s <= 5.0, 'station shows the real job within 5 s of it finishing, no reload', current.update_lag_s)
  const res = (await api('GET', `/api/line/jobs/${job.job_id}/result`)).json
  const fmt = c => `${c.kind}${c.side ? `/${c.side}` : ''}:${c.state}/${c.decision}${typeof c.actual === 'number' ? `=${c.actual.toFixed(2)}` : ''}${typeof c.expected === 'number' ? `(nom ${c.expected.toFixed(2)})` : ''}`
  current.engine = res && {
    validity: res.inspection_validity && res.inspection_validity.state, stage: res.stage_decision && res.stage_decision.status,
    inferred_stage: res.stage_decision && res.stage_decision.inferred_stage, summary: res.summary && res.summary.state,
    bends: (res.bends || []).map(b => ({ label: b.label, seq: b.sequence_index, state: b.state, characteristics: b.characteristics.map(fmt) })),
  }
  check(res && res.bends && res.bends.length === realBends, 'the result reports every recipe bend', res && res.bends && res.bends.length)
  const measured = (res.bends || []).flatMap(b => b.characteristics).filter(c => c.state === 'MEASURED').length
  current.engine.measured_characteristics = measured
  // Engine outcome on real data is engineering-ledger evidence for A1, not an E2E pass criterion.
  note(res.summary.state !== 'INVALID_INSPECTION', 'engine: the real FULL scan gives a valid inspection (informational; engine defects go to A1)', { summary: res.summary.state, validity: res.inspection_validity && res.inspection_validity.state, measured })
  const state = await sattr('[data-testid="summary-banner"]', 'data-summary-state')
  check(!!state && state === res.summary.state, 'verdict banner shows the engine summary state', { banner: state, api: res && res.summary && res.summary.state })
  const rows = await kiosk.locator('tr[data-sequence]').count()
  check(rows === realBends, 'flange-first table has one row per bend', rows)
  await waitFor(async () => ['heatmap', 'neutral'].includes(await sattr('[data-testid="station-viewer"]', 'data-scan-colour')), 30000, 'scan drawn').catch(() => null)
  // The part itself must be on screen: the stage mesh, or the finished part when the stage was not decoded.
  const cadMesh = await waitFor(async () => { const v = await sattr('[data-testid="station-viewer"]', 'data-cad-mesh'); return ['stage', 'final'].includes(v) ? v : null }, 30000, 'a CAD mesh').catch(async () => sattr('[data-testid="station-viewer"]', 'data-cad-mesh'))
  current.cad_mesh = cadMesh
  check(['stage', 'final'].includes(cadMesh), 'the station draws a CAD mesh of the real part (stage model, else the finished part)', cadMesh)
  await sleep(1000)
  const px = await canvasPixels(kiosk)
  current.canvas = px
  // Not blank = surface pixels, not just bend-line wireframe (the synthetic part paints ~1200; lines alone ~400).
  check(px.painted > 1000 && px.colours > 20, 'WebGL canvas shows the part surface, not only bend lines (real scan)', px)
  const rl = labelProblems(await kiosk.evaluate(probeLabels), { minVisible: 1 })
  check(rl.length === 0, 'real part: no overlapping or mis-tagged bend labels, none over the viewer controls', rl)
  // The server ends each event stream after ~5 min; the station must be back on the live stream
  // (not stuck on polling) by the time the real job is shown, minutes after K1 connected.
  const conn = await waitFor(async () => (await sattr('[data-testid="station-root"]', 'data-connection')) === 'live' ? 'live' : null, 5000, 'the live stream')
    .catch(async () => sattr('[data-testid="station-root"]', 'data-connection'))
  current.connection = conn
  note(conn === 'live', 'station is on the live stream when the real job is shown (stream reopened after the server lifetime)', conn)
  await settle(kiosk)
  await shot(kiosk, 'real-station-result', `real FULL scan: banner ${state}; ${rows} table rows; CAD mesh ${cadMesh}; labels clean; connection ${conn}`)
}, { requires: ['R3', 'K1'] })

await step('R5', 'real 10839000: worst-first close-up, station in Hebrew, PDF export (Hebrew + English)', async () => {
  await kiosk.click('[data-action="worst"]')
  await sleep(900)
  const worst = await sattr('[data-testid="station-viewer"]', 'data-selected-bend')
  check(!!worst, 'worst-first selected a bend', worst)
  const wl = labelProblems(await kiosk.evaluate(probeLabels), { selectedUid: worst })
  check(wl.length === 0, 'real part worst-first: the highlighted label is the selected bend, on top, nothing mis-tagged (G4)', wl)
  for (let i = 0; i < 4; i++) { await kiosk.mouse.wheel(0, -120); await sleep(50) }
  await sleep(500)
  const zl = labelProblems(await kiosk.evaluate(probeLabels), { selectedUid: worst, minVisible: 1 })
  check(zl.length === 0, 'real part close-up: labels clean, none over the toolbar', zl)
  await shot(kiosk, 'real-station-worst-first', `selected bend ${worst} (close-up; reviewer judges LANDS / TAG_OK)`)
  await kiosk.click('[data-action="fit"]')
  await kiosk.click('[data-testid="lang-toggle"]')
  await kiosk.locator('[data-testid="station-root"][dir="rtl"]').waitFor()
  await sleep(500)
  const m = await kiosk.evaluate(() => ({ sw: document.documentElement.scrollWidth, sh: document.documentElement.scrollHeight }))
  check(m.sw <= 1920 && m.sh <= 1080, 'RTL station (real part) has no page scroll', m)
  const legend = await kiosk.locator('[data-testid="heatmap-legend"]').innerText()
  check(!/[A-Z]{3,}_|_[A-Z]{3,}|validity|mirrored part suspected/.test(legend), 'heatmap reason in Hebrew operator wording, not an enum', legend)
  const hl = labelProblems(await kiosk.evaluate(probeLabels), { minVisible: 1 })
  check(hl.length === 0, 'RTL real part: labels clean', hl)
  await shot(kiosk, 'real-station-hebrew-rtl', 'station-root dir=rtl with the real job, no overflow')
  await kiosk.click('[data-testid="lang-toggle"]')
  await kiosk.locator('[data-testid="station-root"][dir="ltr"]').waitFor()
  const [download] = await Promise.all([kiosk.waitForEvent('download', { timeout: 60000 }), kiosk.click('[data-testid="download-pdf"]')])
  const outDir = join(OUT, 'pdf', 'real')
  await mkdir(outDir, { recursive: true })
  const pdfPath = join(outDir, 'report.pdf')
  await download.saveAs(pdfPath)
  const r = run(PY, [SUPPORT, 'pdf-pages', '--pdf', pdfPath, '--out', outDir, '--expect', realSku, '--header-ids', realSku, realRecipe],
    { cwd: REPO, env: { ...process.env, PYTHONPATH: `${REPO}:${REPO}/backend` } })
  const doc = JSON.parse(r.out.trim().split('\n').pop() || '{}')
  current.pdf = doc
  check(doc.header && doc.header.pages_wrong.length === 0, 'real-part PDF header keeps the numeric SKU and recipe id in order on every page (bidi)', doc.header)
  check(r.code === 0 && doc.ok && doc.pages.length >= 1, 'real-part PDF rendered to page images and names the SKU', { code: r.code, pages: doc.pages && doc.pages.length, found: doc.expect_found })
  check(/[֐-׿]{3,}/.test(readFileSync(join(outDir, 'pdf_text.txt'), 'utf8')), 'real-part PDF contains Hebrew text')
}, { requires: ['R4'] })

await step('Z1', 'no page errors in the wizard or the station', async () => {
  current.page_errors = pageErrors.slice(0, 20)
  check(pageErrors.length === 0, 'no uncaught page errors', pageErrors.slice(0, 5))
}, {})

// ------------------------------------------------------------------ teardown
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
process.exit(steps.some(s => s.status === 'FAIL') ? 1 : 0)
