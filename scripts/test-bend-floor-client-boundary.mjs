// Source-only synthetic transport/render tests. No browser, backend, account,
// geometry library, credentials, physical input or stored artifact is loaded.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import vm from 'node:vm'

const require = createRequire(new URL('../frontend/react/package.json', import.meta.url))
const ts = require('typescript')
const React = require('react')
const { renderToStaticMarkup } = require('react-dom/server')
const base = new URL('../frontend/react/src/', import.meta.url)
const source = (path) => readFileSync(new URL(path, base), 'utf8')
const sources = Object.fromEntries([
  'services/authenticatedArtifact.ts', 'components/AuthenticatedArtifact.tsx',
  'components/BenderView.tsx', 'components/BendMapDiagram.tsx', 'pages/BenderViewPage.tsx',
  'features/bend-inspection/model/decisionContract.ts',
  'features/bend-inspection/pages/BendInspectionPage.tsx', 'shared/3d/ThreeViewer.tsx',
].map((path) => [path, source(path)]))

function evaluate(path, dependencies = {}, globals = {}, text = sources[path]) {
  const imports = { react: React, 'react/jsx-runtime': require('react/jsx-runtime'), ...dependencies }
  const sandbox = { exports: {}, ...globals, require(name) {
    assert.ok(Object.hasOwn(imports, name), `Unapproved runtime import in ${path}: ${name}`)
    return imports[name]
  } }
  const result = ts.transpileModule(text, {
    fileName: fileURLToPath(new URL(path, base)),
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  })
  vm.runInNewContext(result.outputText, sandbox, { filename: path })
  return { exports: sandbox.exports, sandbox }
}

let checks = 0
const failures = []
async function check(name, run) {
  try { await run(); checks += 1; console.log(`PASS ${name}`) }
  catch (error) { failures.push(name); console.error(`FAIL ${name}: ${error.message.slice(0, 500)}`) }
}
const ORIGIN = 'https://floor.example.test'
const RESOURCE = '/api/bend-inspection/bend_synthetic/overlay/overview.png'
const transportCalls = []
let token = 'synthetic-test-token-not-a-real-credential'
let responseStatus = 200
let networkFailure = null
let responseBlobType = 'image/png'
const transport = evaluate('services/authenticatedArtifact.ts', {}, {
  URL, window: { location: { origin: ORIGIN } },
  localStorage: { getItem(key) { assert.equal(key, 'token'); return token } },
  fetch: async (url, options) => {
    transportCalls.push({ url, options })
    if (networkFailure) throw networkFailure
    return { ok: responseStatus >= 200 && responseStatus < 300, status: responseStatus,
      arrayBuffer: async () => new ArrayBuffer(8), blob: async () => ({ type: responseBlobType }) }
  },
}).exports

for (const path of [RESOURCE, '/api/aligned-scan/bend_synthetic.ply', '/api/reference-mesh/bend_synthetic.ply',
  '/api/heatmap/bend_synthetic/overview.png', '/api/heatmaps/bend_synthetic', '/api/files/bend_synthetic/scan.ply',
  '/api/bend-inspection/bend_synthetic/pdf', `${ORIGIN}${RESOURCE}`]) {
  await check(`allowed same-origin resource: ${path}`, () => assert.equal(new URL(transport.artifactRequestUrl(path, ORIGIN)).origin, ORIGIN))
}
for (const path of [
  `https://foreign.example.test${RESOURCE}`, `//foreign.example.test${RESOURCE}`, `//floor.example.test${RESOURCE}`,
  `https://user:pass@floor.example.test${RESOURCE}`, `https://floor.example.test:444${RESOURCE}`,
  `http://floor.example.test${RESOURCE}`, `${RESOURCE}?token=synthetic`, `${RESOURCE}?access_token=synthetic`,
  `${RESOURCE}#token=synthetic`, 'javascript:alert(1)', 'data:image/png;base64,AA==',
  'blob:https://floor.example.test/synthetic', '/api/auth/me', '/api/bend-inspection-malicious/job',
  '/api/bend-inspection/../../secret', '/assets/image.png', ` ${RESOURCE}`, `${RESOURCE} `,
  '\\\\floor.example.test/api/files/bend_synthetic/scan.ply',
]) {
  await check(`reject unsafe artifact address: ${path}`, () => assert.throws(() => transport.artifactRequestUrl(path, ORIGIN)))
}
await check('bearer is header-only; transport forbids redirects and caching', async () => {
  transportCalls.length = 0
  const controller = new AbortController()
  await transport.fetchAuthenticatedArtifact(RESOURCE, controller.signal)
  assert.equal(transportCalls.length, 1)
  const call = transportCalls[0]
  assert.equal(call.url, `${ORIGIN}${RESOURCE}`)
  assert.ok(!call.url.includes(token))
  assert.equal(call.options.headers.Authorization, `Bearer ${token}`)
  assert.equal(call.options.redirect, 'error')
  assert.equal(call.options.cache, 'no-store')
  assert.equal(call.options.credentials, 'same-origin')
  assert.equal(call.options.signal, controller.signal)
})
await check('missing auth never dispatches a request', async () => {
  const saved = token; token = null; transportCalls.length = 0
  try { await assert.rejects(transport.fetchAuthenticatedArtifact(RESOURCE), /Sign in again/) }
  finally { token = saved }
  assert.equal(transportCalls.length, 0)
})
await check('unsafe URL is rejected before transport', async () => {
  transportCalls.length = 0
  await assert.rejects(transport.fetchAuthenticatedArtifact(`https://foreign.example.test${RESOURCE}`), /not an allowed/)
  assert.equal(transportCalls.length, 0)
})
for (const [status, expected] of [[401, /session expired/], [403, /unavailable for this account/],
  [404, /unavailable for this account/], [500, /Retry or reopen/]]) {
  await check(`HTTP ${status} has conservative recoverable feedback`, async () => {
    responseStatus = status
    try { await assert.rejects(transport.fetchAuthenticatedArtifact(RESOURCE), expected) }
    finally { responseStatus = 200 }
  })
}
await check('network/redirect failure is not interpreted as an inspection result', async () => {
  networkFailure = new TypeError('Synthetic fetch or redirect failure')
  try { await assert.rejects(transport.fetchAuthenticatedArtifact(RESOURCE), /Synthetic fetch or redirect failure/) }
  finally { networkFailure = null }
})

const decisions = evaluate('features/bend-inspection/model/decisionContract.ts').exports
const diagrams = evaluate('components/BendMapDiagram.tsx').exports
const artifacts = evaluate('components/AuthenticatedArtifact.tsx', { '../services/authenticatedArtifact': transport }).exports
const icon = (props) => React.createElement('svg', { ...props, 'aria-hidden': true })
const icons = { Printer: icon, Box: icon, BarChart3: icon, ArrowLeft: icon }
const bender = evaluate('components/BenderView.tsx', {
  clsx: require('clsx'), 'lucide-react': icons,
  './BendMapDiagram': diagrams, './AuthenticatedArtifact': artifacts,
  '../features/bend-inspection/model/decisionContract': decisions,
}).exports.default
const match = {
  bend_id: 'CAD_SYNTHETIC', status: 'PASS', physical_completion_state: 'FORMED',
  target_angle: 90, tolerance_angle: 1, measured_angle: 89.4, angle_deviation: -0.6,
  cad_line_start: [0, 0, 0], cad_line_end: [1, 0, 0], action_item: 'ADJUST MACHINE NOW',
  correction_angle: 11, machine_correction: 'ACCEPT THIS PART',
}
for (const status of ['PASS', 'FAIL', 'WARNING']) {
  for (const physical of ['FORMED', 'NOT_FORMED', 'PARTIALLY_FORMED']) {
    await check(`secondary view neutralizes raw ${status}/${physical}`, () => {
      const html = renderToStaticMarkup(React.createElement(bender, {
        matches: [{ ...match, status, physical_completion_state: physical }], jobId: 'bend_synthetic',
      }))
      assert.match(html, /not qualified for production accept\/reject/)
      assert.match(html, /no machine adjustment is authorized/)
      assert.match(html, /State unknown/)
      assert.match(html, /Observed count unavailable/)
      assert.doesNotMatch(html, /ADJUST MACHINE NOW|ACCEPT THIS PART|correction_angle/)
      assert.doesNotMatch(html, /#34d399|#f87171|bg-emerald|bg-red|text-emerald|text-red/)
      assert.match(html, /Experimental measurements; tolerance disposition remains unverified/)
    })
  }
}
await check('image SSR exposes loading state, never raw protected src', () => {
  const html = renderToStaticMarkup(React.createElement(artifacts.AuthenticatedArtifactImage, { src: RESOURCE, alt: 'Synthetic view' }))
  assert.match(html, /role="status"/)
  assert.match(html, /Loading inspection image/)
  assert.doesNotMatch(html, /<img|token|src=/)
})
await check('download SSR uses button, not uncredentialed protected anchor', () => {
  const html = renderToStaticMarkup(React.createElement(artifacts.AuthenticatedArtifactDownload, {
    url: '/api/bend-inspection/bend_synthetic/pdf', filename: 'Synthetic.pdf',
  }))
  assert.match(html, /<button/)
  assert.match(html, /Download PDF/)
  assert.doesNotMatch(html, /<a|href=|token/)
})

function hookHarness() {
  const state = [], previousDeps = [], pending = []
  let cursor = 0
  const hooks = {
    useState(initial) {
      const index = cursor++
      if (!(index in state)) state[index] = initial
      return [state[index], (value) => { state[index] = typeof value === 'function' ? value(state[index]) : value }]
    },
    useEffect(effect, deps) {
      const index = cursor++
      if (!previousDeps[index] || deps.some((value, i) => value !== previousDeps[index][i])) pending.push(effect)
      previousDeps[index] = deps
    },
  }
  return { hooks, render(Component, props) { cursor = 0; return Component(props) },
    effects() { return pending.splice(0).map((effect) => effect()) } }
}
const settle = () => new Promise((resolve) => setImmediate(resolve))
for (const status of [401, 403, 404, 500]) {
  await check(`image effect HTTP ${status} offers retry without defect wording`, async () => {
    const hooks = hookHarness()
    const component = evaluate('components/AuthenticatedArtifact.tsx', {
      react: hooks.hooks, '../services/authenticatedArtifact': transport,
    }, { AbortController, URL: { createObjectURL() { assert.fail('Error response created image URL') } } }).exports.AuthenticatedArtifactImage
    responseStatus = status
    const props = { src: RESOURCE, alt: 'Synthetic view' }
    try {
      hooks.render(component, props); const cleanup = hooks.effects(); await settle()
      const html = renderToStaticMarkup(hooks.render(component, props))
      assert.match(html, /role="alert"/)
      assert.match(html, /Retry image/)
      assert.match(html, /does not establish a part defect/)
      cleanup.forEach((fn) => fn?.())
    } finally { responseStatus = 200 }
  })
}
await check('image effect accepts raster MIME and revokes object URL on cleanup', async () => {
  const hooks = hookHarness(), revoked = []
  const component = evaluate('components/AuthenticatedArtifact.tsx', {
    react: hooks.hooks, '../services/authenticatedArtifact': transport,
  }, { AbortController, URL: { createObjectURL: () => 'blob:synthetic-local', revokeObjectURL: (url) => revoked.push(url) } }).exports.AuthenticatedArtifactImage
  const props = { src: RESOURCE, alt: 'Synthetic view' }
  hooks.render(component, props); const cleanup = hooks.effects(); await settle()
  const html = renderToStaticMarkup(hooks.render(component, props))
  assert.match(html, /src="blob:synthetic-local"/)
  assert.match(html, /alt="Synthetic view"/)
  cleanup.forEach((fn) => fn?.())
  assert.deepEqual(revoked, ['blob:synthetic-local'])
})
await check('image effect rejects active HTML/SVG payloads', async () => {
  for (const type of ['text/html', 'image/svg+xml']) {
    responseBlobType = type
    const hooks = hookHarness()
    const component = evaluate('components/AuthenticatedArtifact.tsx', {
      react: hooks.hooks, '../services/authenticatedArtifact': transport,
    }, { AbortController, URL: { createObjectURL() { assert.fail('Unsafe MIME created object URL') } } }).exports.AuthenticatedArtifactImage
    const props = { src: RESOURCE, alt: 'Synthetic view' }
    try {
      hooks.render(component, props); const cleanup = hooks.effects(); await settle()
      assert.match(renderToStaticMarkup(hooks.render(component, props)), /not a supported inspection image/)
      cleanup.forEach((fn) => fn?.())
    } finally { responseBlobType = 'image/png' }
  }
})

await check('download is disabled while pending and uses only a transient local blob link', async () => {
  const hooks = hookHarness(), events = [], timers = []
  let resolveResponse
  const pending = new Promise((resolve) => { resolveResponse = resolve })
  const anchor = { href: null, download: null, click() { events.push(['click', this.href, this.download]) }, remove() { events.push(['remove']) } }
  const component = evaluate('components/AuthenticatedArtifact.tsx', {
    react: hooks.hooks, '../services/authenticatedArtifact': { fetchAuthenticatedArtifact(url) {
      events.push(['fetch', url]); return pending
    } },
  }, {
    URL: { createObjectURL: () => 'blob:synthetic-download', revokeObjectURL: (url) => events.push(['revoke', url]) },
    document: { createElement(tag) { assert.equal(tag, 'a'); return anchor }, body: { appendChild() {} } },
    setTimeout: (callback) => timers.push(callback),
  }).exports.AuthenticatedArtifactDownload
  const props = { url: '/api/bend-inspection/bend_synthetic/pdf', filename: 'Synthetic.pdf' }
  const tree = hooks.render(component, props)
  React.Children.toArray(tree.props.children).find((child) => child.type === 'button').props.onClick()
  const loading = renderToStaticMarkup(hooks.render(component, props))
  assert.match(loading, /disabled=""/)
  assert.match(loading, /Preparing download/)
  resolveResponse({ blob: async () => ({ type: 'application/pdf' }) }); await settle()
  const completed = renderToStaticMarkup(hooks.render(component, props))
  assert.doesNotMatch(completed, /disabled=""/)
  assert.deepEqual(events[0], ['fetch', props.url])
  assert.deepEqual(events[1], ['click', 'blob:synthetic-download', 'Synthetic.pdf'])
  timers.forEach((callback) => callback())
  assert.ok(events.some((event) => event[0] === 'revoke'))
})

for (const status of [401, 403, 404, 500]) {
  await check(`download HTTP ${status} renders an alert and does not create a link`, async () => {
    const hooks = hookHarness()
    const component = evaluate('components/AuthenticatedArtifact.tsx', {
      react: hooks.hooks, '../services/authenticatedArtifact': transport,
    }, { URL: { createObjectURL() { assert.fail('Error response created download link') } } }).exports.AuthenticatedArtifactDownload
    const props = { url: '/api/bend-inspection/bend_synthetic/pdf', filename: 'Synthetic.pdf' }
    responseStatus = status
    try {
      const tree = hooks.render(component, props)
      React.Children.toArray(tree.props.children).find((child) => child.type === 'button').props.onClick()
      await settle()
      const html = renderToStaticMarkup(hooks.render(component, props))
      assert.match(html, /role="alert"/)
      assert.doesNotMatch(html, /disabled=""|<a/)
    } finally { responseStatus = 200 }
  })
}

for (const rawContract of [undefined, { schema_version: 1, production_disposition: 'ACCEPT' }]) {
  await check(`secondary page validates ${rawContract ? 'malformed' : 'missing'} decision contract before rendering`, () => {
    let received = null
    const job = { part_id: 'Synthetic', report: { matches: [match], summary: { completed_bends: 99 }, decision_contract: rawContract } }
    const page = evaluate('pages/BenderViewPage.tsx', {
      'react-router-dom': { useParams: () => ({ jobId: 'bend_synthetic' }), Link: ({ to, children, ...props }) => React.createElement('a', { ...props, href: to }, children) },
      '@tanstack/react-query': { useQuery: () => ({ data: job, isLoading: false, error: null }) },
      '../services/api': { bendInspectionApi: { get() { assert.fail('Query executed network') } } },
      '../components/BenderView': { default: (props) => { received = props; return React.createElement(bender, props) }, __esModule: true },
      'lucide-react': icons, '../features/bend-inspection/model/decisionContract': decisions,
    }).exports.default
    const html = renderToStaticMarkup(React.createElement(page))
    assert.equal(received.matches[0].decision_state, 'UNKNOWN')
    assert.equal(received.matches[0].status, 'NOT_DETECTED')
    assert.equal(received.observedFormedRatio, null)
    assert.match(html, /Observed count unavailable/)
    assert.doesNotMatch(html, /99\/|ADJUST MACHINE NOW|ACCEPT THIS PART/)
  })
}
await check('main page presentation invokes validated decision helper', () => {
  const text = sources['features/bend-inspection/pages/BendInspectionPage.tsx']
  assert.match(text, /presentBendDecision\(report\)/)
  assert.match(text, /presentBendDecision\(job\.report\)/)
  assert.match(text, /decision\.observedFormedRatio/)
})
await check('3D local binary transport is credentialed and remote HDR is absent', () => {
  const text = sources['shared/3d/ThreeViewer.tsx']
  assert.match(text, /fetchAuthenticatedArtifact\(url(?:, controller.signal)?\)/)
  assert.match(text, /loader\.parse\(binary\)/)
  assert.doesNotMatch(text, /<Environment|preset="warehouse"|\.hdr["']/)
})

// Execute the real nested loader functions, with parser/loader objects replaced.
// This verifies branching and transport without importing Three.js or rendering.
const viewerAst = ts.createSourceFile('ThreeViewer.tsx', sources['shared/3d/ThreeViewer.tsx'], ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
function initializer(name) {
  let found
  function visit(node) {
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.name.text === name) found = node.initializer.getText(viewerAst)
    ts.forEachChild(node, visit)
  }
  visit(viewerAst)
  assert.ok(found, `Source function ${name} is present`)
  return found
}
const parserCalls = []
class SyntheticLoader {
  parse(binary) { parserCalls.push(['parse', binary.byteLength]); return { index: null } }
  load(url, resolve) { parserCalls.push(['load', url]); resolve({ index: null }) }
}
const meshController = new AbortController()
const mesh = evaluate('synthetic-extracted-viewer.ts', {}, {
  URL, window: { location: { origin: ORIGIN } }, controller: meshController,
  STLLoader: SyntheticLoader, PLYLoader: SyntheticLoader,
  fetchAuthenticatedArtifact: transport.fetchAuthenticatedArtifact,
}, `export const detectGeometryType = ${initializer('detectGeometryType')};\nexport const loadGeometry = ${initializer('loadGeometry')};`).exports
for (const path of ['/api/aligned-scan/bend_synthetic.ply', '/api/reference-mesh/bend_synthetic.ply', '/api/files/bend_synthetic/cad.stl']) {
  await check(`3D ${path} uses one authenticated fetch then binary parser`, async () => {
    parserCalls.length = 0; transportCalls.length = 0
    await mesh.loadGeometry(path)
    assert.deepEqual(parserCalls, [['parse', 8]])
    assert.equal(transportCalls.length, 1)
    assert.equal(transportCalls[0].options.signal, meshController.signal)
    assert.equal(transportCalls[0].options.headers.Authorization, `Bearer ${token}`)
  })
}
for (const path of ['https://foreign.example.test/scan.ply', '//floor.example.test/api/aligned-scan/bend_synthetic.ply',
  '/api/auth/me.ply', '/api/aligned-scan/bend_synthetic.ply?token=synthetic',
  'https://user:pass@floor.example.test/api/aligned-scan/bend_synthetic.ply']) {
  await check(`3D rejects unsafe source before any parser or transport: ${path}`, async () => {
    parserCalls.length = 0; transportCalls.length = 0
    await assert.rejects(mesh.loadGeometry(path))
    assert.equal(parserCalls.length, 0)
    assert.equal(transportCalls.length, 0)
  })
}

console.log(`${checks} synthetic floor-client checks passed; ${failures.length} failed`)
if (failures.length) process.exitCode = 1
