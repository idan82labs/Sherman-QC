// Disposable loopback-only compiled-UI harness. All APIs and binary geometry
// are invented here. No backend, proxy, account store or research inputs.
// Run after npm --prefix frontend/react run build. Stop with Ctrl-C.
import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { dirname, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const dist = resolve(dirname(fileURLToPath(import.meta.url)), '../frontend/dist')
const token = 'SYNTHETIC_UI_ONLY_NOT_A_CREDENTIAL'
const user = { id: 1, username: 'synthetic-floor', role: 'operator', is_active: true,
  email: 'synthetic@example.invalid', created_at: '2026-09-07T00:00:00Z' }
const jobId = 'bend_synthetic_floor_ui'
const prefix = `/api/bend-inspection/${jobId}`
const base = { bend_form: 'FOLDED', countable_in_regression: true, target_angle: 90, target_radius: 2,
  measured_angle: 89, measured_radius: 2, angle_deviation: -1, radius_deviation: 0,
  confidence: 1, tolerance_angle: 1, tolerance_radius: 0.5,
  action_item: 'UNSAFE_RAW_MACHINE_ADJUSTMENT_MUST_NOT_APPEAR' }
const job = { job_id: jobId, part_id: 'SYNTHETIC FLOOR CHECK', part_name: 'Not a physical part',
  status: 'completed', progress: 100, created_at: '2026-09-07T00:00:00Z', processing_time_ms: 1,
  artifacts: { reference_mesh_url: `/api/reference-mesh/${jobId}.ply`,
    bend_overlay_overview_url: `${prefix}/overlay/overview.png`, bend_report_pdf_url: `${prefix}/pdf`,
    bender_view_front_url: `${prefix}/overlay/front.png` },
  report: { part_id: 'SYNTHETIC FLOOR CHECK', processing_time_ms: 1, unmatched_detections: [],
    summary: { total_bends: 999, expected_bends: 999, completed_bends: 999, detected: 999, passed: 999,
      failed: 0, warnings: 0, progress_pct: 100, structured_completed_bends: 9 },
    matches: [{ ...base, bend_id: 'A', status: 'PASS', physical_completion_state: 'FORMED' },
      { ...base, bend_id: 'B', status: 'FAIL', physical_completion_state: 'NOT_FORMED' },
      { ...base, bend_id: 'C', status: 'WARNING', physical_completion_state: 'PARTIALLY_FORMED' }],
    decision_contract: { schema_version: 1, kind: 'BEND_DECISION_CONTRACT_V1',
      observed_counts: { formed: 1, unformed: 0, unknown: 2, total: 3, partially_observed: 1 },
      registration: 'UNVERIFIED', render_evidence: 'UNAVAILABLE', production_disposition: 'NOT_AUTHORIZED', reasons: [],
      bends: [{ bend_id: 'A', state: 'FORMED', observation_detail: 'OBSERVED_FORMED', reasons: [] },
        { bend_id: 'B', state: 'UNKNOWN', observation_detail: 'UNKNOWN', reasons: [] },
        { bend_id: 'C', state: 'UNKNOWN', observation_detail: 'PARTIALLY_OBSERVED', reasons: [] }] } } }
const readiness = { schema_version: 1, kind: 'SHERMAN_QC_BEND_INSPECTION_READINESS', status: 'PROMOTION_BLOCKED',
  operating_mode: 'RESEARCH_ONLY', production_use_allowed: false, physical_generalization_progress: false,
  active_blockers: ['SYNTHETIC_UI_ONLY'], generalization_progress_indicator: {}, lane_status: {} }
const labels = { schema_version: 1, kind: 'sherman_qc_bend_instance_labels', job_id: jobId,
  coordinate_frame: 'reference_mesh_raw', review_status: 'in_progress', manual_bends: [], marker_reviews: [] }
const json = new Map([['/api/auth/me', user], ['/api/bend-inspection/readiness', readiness],
  ['/api/bend-inspection/jobs', { jobs: [job], total: 1 }], [prefix, job], [`${prefix}/labels`, labels],
  ['/api/jobs', { jobs: [], total: 0 }], ['/api/stats', { total_jobs: 0, completed_jobs: 0, failed_jobs: 0,
    processing_jobs: 0, pending_jobs: 0, pass_rate: 0, avg_processing_time: 0, jobs_today: 0, total_parts: 0,
    recent_jobs: [], status_distribution: {}, daily_stats: [] }]])
const ply = 'ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n3 0 1 2\n3 1 3 2\n'
// One invented transparent PNG pixel; no photograph or physical data.
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg==', 'base64')
const metrics = { acceptedApiRequests: 0, deniedApiRequests: 0, binaryRequests: {}, unexpectedPaths: [] }
let frontAttempts = 0
const server = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://127.0.0.1').pathname
  const send = (status, body, contentType = 'application/json') => {
    res.writeHead(status, { 'Content-Type': contentType, 'Cache-Control': 'no-store',
      'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; font-src 'self'; object-src 'none'; frame-src 'none'; base-uri 'none'" })
    res.end(typeof body === 'object' && !Buffer.isBuffer(body) ? JSON.stringify(body) : body)
  }
  try {
    if (path === '/__synthetic_metrics' && req.method === 'GET') { send(200, metrics); return }
    if (path === '/api/auth/login' && req.method === 'POST') {
      let length = 0
      for await (const chunk of req) { length += chunk.length; if (length > 4096) { send(413, {}); return } }
      send(200, { access_token: token, token_type: 'bearer', user }); return
    }
    if (path.startsWith('/api/')) {
      if (req.method !== 'GET' || req.headers.authorization !== `Bearer ${token}`) {
        metrics.deniedApiRequests += 1; send(401, { detail: 'Synthetic harness authentication required' }); return
      }
      metrics.acceptedApiRequests += 1
      if (json.has(path)) { send(200, json.get(path)); return }
      metrics.binaryRequests[path] = (metrics.binaryRequests[path] ?? 0) + 1
      if (path === `/api/aligned-scan/${jobId}.ply` || path === `/api/reference-mesh/${jobId}.ply`) { send(200, ply, 'application/x-ply'); return }
      if (path === `${prefix}/overlay/overview.png`) { send(200, png, 'image/png'); return }
      if (path === `${prefix}/overlay/front.png`) {
        // One controlled failure exercises the visible retry path.
        frontAttempts += 1
        if (frontAttempts === 1) { send(503, { detail: 'Synthetic retry exercise' }); return }
        send(200, png, 'image/png'); return
      }
      if (path === `${prefix}/pdf`) { send(503, { detail: 'Synthetic PDF unavailable' }); return }
      metrics.unexpectedPaths.push(path); send(404, {}); return
    }
    if (req.method !== 'GET') { send(405, {}); return }
    const file = path.startsWith('/assets/') ? resolve(dist, `.${path}`) : join(dist, 'index.html')
    if (!file.startsWith(`${dist}${sep}`)) { send(403, {}); return }
    const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' }
    send(200, await readFile(file), types[extname(file)] ?? 'application/octet-stream')
  } catch { send(404, {}) }
})
server.listen(0, '127.0.0.1', () => console.log(JSON.stringify({
  kind: 'SYNTHETIC_UI_ONLY', origin: `http://127.0.0.1:${server.address().port}`,
  username: 'synthetic-floor', password: 'synthetic-not-a-password', jobId,
})))
