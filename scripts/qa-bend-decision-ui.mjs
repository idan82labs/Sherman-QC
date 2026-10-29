// Compiled-UI QA using only inline synthetic data and freshly built frontend/dist.
// No real auth, backend, user browser profile, artifact corpus or external network.
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { readFile, mkdtemp } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const require = createRequire(new URL('../frontend/react/package.json', import.meta.url))
const { chromium } = require('playwright')
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const dist = join(root, 'frontend', 'dist')
const user = { id: 1, username: 'Synthetic QA', email: 'qa@example.invalid', role: 'operator', is_active: true, created_at: '2026-09-07T00:00:00Z', last_login: null }
const baseMatch = { bend_form: 'FOLDED', countable_in_regression: true, target_angle: 90, target_radius: 2,
  measured_angle: 90, measured_radius: 2, angle_deviation: 0, radius_deviation: 0,
  confidence: 1, tolerance_angle: 1, tolerance_radius: 0.5, action_item: 'UNSAFE_RAW_ACTION_MUST_NOT_DISPLAY' }
const job = {
  job_id: 'SYNTHETIC_DECISION_QA', part_id: 'SYNTHETIC DECISION CHECK', part_name: 'Inline fixture only',
  status: 'completed', progress: 100, created_at: '2026-09-07T00:00:00Z', processing_time_ms: 1,
  report: {
    part_id: 'SYNTHETIC DECISION CHECK', processing_time_ms: 1, unmatched_detections: [],
    summary: { total_bends: 999, expected_bends: 999, completed_bends: 999, detected: 999, passed: 999,
      failed: 0, warnings: 1, progress_pct: 100, structured_completed_bends: 9, structured_count_delta_vs_match: -990 },
    matches: [
      { ...baseMatch, bend_id: 'A', status: 'PASS', observability_state: 'UNKNOWN', physical_completion_state: 'FORMED' },
      { ...baseMatch, bend_id: 'B', status: 'PASS', observability_state: 'FORMED', physical_completion_state: 'FORMED' },
      { ...baseMatch, bend_id: 'C', status: 'WARNING', observability_state: 'UNKNOWN', observability_detail_state: 'PARTIALLY_OBSERVED', physical_completion_state: 'PARTIALLY_FORMED' },
    ],
    decision_contract: {
      schema_version: 1, kind: 'BEND_DECISION_CONTRACT_V1',
      observed_counts: { formed: 1, unformed: 0, unknown: 2, total: 3, partially_observed: 1 },
      registration: 'UNVERIFIED', render_evidence: 'UNAVAILABLE', production_disposition: 'NOT_AUTHORIZED', reasons: ['SYNTHETIC_QA_ONLY'],
      bends: [
        { bend_id: 'A', state: 'UNKNOWN', observation_detail: 'UNKNOWN', reasons: ['INCOMPLETE_OBSERVATION'] },
        { bend_id: 'B', state: 'FORMED', observation_detail: 'OBSERVED_FORMED', reasons: [] },
        { bend_id: 'C', state: 'UNKNOWN', observation_detail: 'PARTIALLY_OBSERVED', reasons: ['PHYSICAL_STAGE_NOT_QUALIFIED'] },
      ],
    },
  },
}
const readiness = { schema_version: 1, kind: 'SHERMAN_QC_BEND_INSPECTION_READINESS', status: 'PROMOTION_BLOCKED',
  operating_mode: 'RESEARCH_ONLY', production_use_allowed: false, physical_generalization_progress: false,
  active_blockers: ['SYNTHETIC_QA_ONLY'], generalization_progress_indicator: { cad_assisted: 'synthetic UI only' }, lane_status: {} }
const labels = { schema_version: 1, kind: 'sherman_qc_bend_instance_labels', job_id: job.job_id,
  coordinate_frame: 'reference_mesh_raw', review_status: 'in_progress', manual_bends: [], marker_reviews: [] }
const jsonResponses = new Map([
  ['/api/auth/me', user], ['/api/bend-inspection/readiness', readiness],
  ['/api/bend-inspection/jobs', { jobs: [job], total: 1 }],
  [`/api/bend-inspection/${job.job_id}`, job], [`/api/bend-inspection/${job.job_id}/labels`, labels],
])
const syntheticPly = 'ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n0 1 0\n'
const contentTypes = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' }
const server = createServer(async (request, response) => {
  try {
    const path = new URL(request.url, 'http://127.0.0.1').pathname
    // Static assets only; this server has no API/proxy and cannot read beyond dist.
    if (request.method !== 'GET' || path.startsWith('/api/')) { response.writeHead(403); response.end(); return }
    const file = path.startsWith('/assets/') ? resolve(dist, `.${path}`) : join(dist, 'index.html')
    if (!file.startsWith(`${dist}${sep}`)) { response.writeHead(403); response.end(); return }
    response.writeHead(200, { 'Content-Type': contentTypes[extname(file)] ?? 'application/octet-stream' })
    response.end(await readFile(file))
  } catch { response.writeHead(404); response.end() }
})
let browser
const results = []
try {
  await new Promise((done) => server.listen(0, '127.0.0.1', done))
  const origin = `http://127.0.0.1:${server.address().port}`
  browser = await chromium.launch({ headless: true, args: ['--disable-background-networking', '--disable-component-update'] })
  const outputDir = await mkdtemp(join(tmpdir(), 'sherman-decision-ui-'))
  for (const [name, viewport] of [['desktop', { width: 1440, height: 1000 }], ['mobile', { width: 390, height: 844 }]]) {
    const context = await browser.newContext({ viewport, serviceWorkers: 'block', reducedMotion: 'reduce' })
    const blocked = [], unexpected = [], errors = []
    await context.routeWebSocket('**/*', (socket) => socket.close())
    await context.route('**/*', async (route) => {
      const request = route.request()
      const url = new URL(request.url())
      if (url.origin !== origin) { blocked.push(request.url()); await route.abort(); return }
      if (url.pathname.startsWith('/api/')) {
        if (request.method() !== 'GET') { unexpected.push(`${request.method()} ${url.pathname}`); await route.abort(); return }
        if (jsonResponses.has(url.pathname)) { await route.fulfill({ json: jsonResponses.get(url.pathname) }); return }
        if (url.pathname === `/api/aligned-scan/${job.job_id}.ply`) { await route.fulfill({ contentType: 'application/x-ply', body: syntheticPly }); return }
        unexpected.push(url.pathname); await route.abort(); return
      }
      // Every browser HTTP request passes this allowlist; only static localhost
      // assets are forwarded, never API requests or foreign origins.
      if (request.method() === 'GET' && (url.pathname === '/bend-inspection' || url.pathname.startsWith('/assets/') || url.pathname === '/favicon.ico')) {
        await route.continue(); return
      }
      unexpected.push(url.pathname); await route.abort()
    })
    await context.addInitScript(({ syntheticUser }) => {
      localStorage.setItem('token', 'SYNTHETIC_UI_QA_NOT_A_CREDENTIAL')
      localStorage.setItem('user', JSON.stringify(syntheticUser))
    }, { syntheticUser: user })
    const page = await context.newPage()
    page.setDefaultTimeout(5000)
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto(`${origin}/bend-inspection`, { waitUntil: 'networkidle', timeout: 20000 })
    const partTitle = page.getByText(job.part_id, { exact: true })
    const titleBox = await partTitle.boundingBox()
    assert.ok(titleBox && titleBox.width > 40, 'job identity must remain visible at this viewport')
    const overview = join(outputDir, `${name}-overview.png`)
    await page.screenshot({ path: overview, fullPage: true })
    console.log(JSON.stringify({ stage: 'overview', name, screenshot: overview, titleBox }))
    await partTitle.click()
    const notice = page.getByRole('note').filter({ hasText: 'Research evidence only' })
    await notice.waitFor({ state: 'visible', timeout: 20000 })
    await page.getByText('Model minus observed', { exact: true }).waitFor()
    const deltaText = await page.getByText('Model minus observed', { exact: true }).locator('..').innerText()
    assert.match(deltaText, /\b8\b/)
    const body = await page.locator('body').innerText()
    assert.match(body, /1\/3/)
    assert.doesNotMatch(body, /1\/999|0\/999|UNSAFE_RAW_ACTION_MUST_NOT_DISPLAY|Not yet/)
    assert.match(body, /State unknown · review/)
    assert.match(body, /Partly observed · unknown/)
    assert.match(body, /Registration: unverified/)
    assert.match(body, /Render evidence: unavailable/)
    // Lighting is now local: no external HDR request is needed for the viewer.
    await page.locator('canvas').waitFor({ state: 'visible', timeout: 10000 })
    assert.deepEqual(blocked, [])
    assert.deepEqual(unexpected, [])
    assert.deepEqual(errors, [])
    const screenshot = join(outputDir, `${name}.png`)
    await page.screenshot({ path: screenshot, fullPage: true })
    assert.equal(await notice.isVisible(), true)
    assert.deepEqual(errors, [])
    const overflow = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth }))
    assert.ok(overflow.document <= overflow.viewport, 'report must not force page-level horizontal scrolling')
    results.push({ name, screenshot, overview, titleBox, overflow, blockedForeignRequests: blocked.length, unexpectedApiRequests: unexpected.length, pageErrors: errors.length })
    await context.close()
  }
  console.log(JSON.stringify({ result: 'PASS_SYNTHETIC_COMPILED_UI_ONLY', results }, null, 2))
} finally {
  if (browser) await browser.close()
  await new Promise((done) => server.close(done))
}
