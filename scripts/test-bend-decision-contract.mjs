// Synthetic, source-only test: no browser, server, artifacts or API imports.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'
import vm from 'node:vm'

const require = createRequire(new URL('../frontend/react/package.json', import.meta.url))
const ts = require('typescript')
const sourcePath = new URL('../frontend/react/src/features/bend-inspection/model/decisionContract.ts', import.meta.url)
const compiled = ts.transpileModule(readFileSync(sourcePath, 'utf8'), {
  fileName: fileURLToPath(sourcePath),
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
const sandbox = { exports: {} }
vm.runInNewContext(compiled, sandbox, { filename: fileURLToPath(sourcePath) })
const { presentBendDecision, decisionLabel, isDecisionCountableBend } = sandbox.exports
const annotationPath = new URL('../frontend/react/src/features/bend-inspection/model/annotationMapper.ts', import.meta.url)
const annotationSandbox = { exports: {}, require: (name) => {
  assert.equal(name, './decisionContract')
  return sandbox.exports
} }
vm.runInNewContext(ts.transpileModule(readFileSync(annotationPath, 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText, annotationSandbox, { filename: fileURLToPath(annotationPath) })
const clone = (value) => JSON.parse(JSON.stringify(value))
const report = {
  summary: { total_bends: 3, completed_bends: 3, detected: 3, structured_completed_bends: 9 },
  matches: [
    { bend_id: 'B1', status: 'PASS', physical_completion_state: 'FORMED', measured_angle: 90 },
    { bend_id: 'B2', status: 'FAIL', physical_completion_state: 'NOT_FORMED', measured_angle: 180 },
    { bend_id: 'B3', status: 'WARNING', physical_completion_state: 'PARTIALLY_FORMED', measured_angle: 120 },
    { bend_id: 'P1', status: 'PASS', countable_in_regression: false },
  ],
  decision_contract: {
    schema_version: 1, kind: 'BEND_DECISION_CONTRACT_V1',
    observed_counts: { formed: 1, unformed: 1, unknown: 1, total: 3, partially_observed: 1 },
    registration: 'UNVERIFIED', render_evidence: 'UNVERIFIED', production_disposition: 'NOT_AUTHORIZED', reasons: [],
    bends: [
      { bend_id: 'B1', state: 'FORMED', observation_detail: 'OBSERVED_FORMED', reasons: [] },
      { bend_id: 'B2', state: 'UNFORMED', observation_detail: 'OBSERVED_NOT_FORMED', reasons: [] },
      { bend_id: 'B3', state: 'UNKNOWN', observation_detail: 'PARTIALLY_OBSERVED', reasons: [] },
      { bend_id: 'P1', state: 'FORMED', observation_detail: 'OBSERVED_FORMED', reasons: [] },
    ],
  },
}
let checks = 0
function check(name, run) { run(); checks += 1; console.log(`PASS ${name}`) }
check('counts are direct evidence; estimate stays secondary', () => {
  const result = presentBendDecision(report)
  assert.equal(result.available, true)
  assert.equal(result.counts.formed, 1)
  assert.equal(result.counts.unformed, 1)
  assert.equal(result.counts.unknown, 1)
  assert.equal(result.modelEstimate, 9)
  assert.equal(result.observedFormedRatio, '1/3')
})
check('observed count denominator cannot be inflated by legacy or CAD totals', () => {
  const changed = clone(report)
  changed.summary.expected_bends = 999
  changed.summary.total_bends = 999
  changed.summary.completed_bends = 999
  assert.equal(presentBendDecision(changed).observedFormedRatio, '1/3')
})
check('no calibrated decision means no PASS/FAIL viewer colors', () => {
  assert.ok(presentBendDecision(report).matches.every((match) => match.status === 'NOT_DETECTED'))
})
check('raw payload is immutable; measurements and physical partial state survive', () => {
  const before = JSON.stringify(report)
  const result = presentBendDecision(report)
  assert.equal(result.matches[2].physical_completion_state, 'PARTIALLY_FORMED')
  assert.equal(result.matches[2].measured_angle, 120)
  assert.equal(result.matches[2].decision_state, 'UNKNOWN')
  assert.equal(JSON.stringify(report), before)
})
check('missing contract never infers evidence from raw PASS or count', () => {
  const legacy = clone(report); delete legacy.decision_contract
  const result = presentBendDecision(legacy)
  assert.equal(result.available, false)
  assert.equal(result.counts.formed, 0)
  assert.equal(result.counts.unknown, 3)
  assert.equal(result.renderEvidence, 'UNAVAILABLE')
  assert.equal(result.observedFormedRatio, null)
  assert.equal(decisionLabel(result.matches[0]), 'State unknown · review')
})
check('malformed, contradictory and future contracts fail closed', () => {
  for (const mutate of [
    (c) => { c.schema_version = 2 },
    (c) => { c.production_disposition = 'ACCEPT' },
    (c) => { c.registration = 'VALID' },
    (c) => { c.registration = ['ABSTAINED'] },
    (c) => { c.render_evidence = ['UNVERIFIED'] },
    (c) => { c.observed_counts.formed = -1 },
    (c) => { c.observed_counts.unknown = NaN },
    (c) => { c.observed_counts.formed = 2 },
    (c) => { c.bends[0].observation_detail = 'UNOBSERVED' },
    (c) => { c.bends[0].observation_detail = 'UNFORMED' },
    (c) => { c.bends[0].observation_detail = 'garbage' },
    (c) => { c.bends[0].state = 'PARTIALLY_FORMED' },
    (c) => { c.bends[0].state = ['FORMED'] },
    (c) => { c.bends.push(c.bends[0]) },
  ]) {
    const changed = clone(report); mutate(changed.decision_contract)
    assert.equal(presentBendDecision(changed).available, false)
  }
})
check('registration abstention cannot coexist with formed evidence', () => {
  const changed = clone(report); changed.decision_contract.registration = 'ABSTAINED'
  assert.equal(presentBendDecision(changed).available, false)
})
check('valid abstention and missing render evidence stay explicit', () => {
  const changed = clone(report)
  changed.decision_contract.registration = 'ABSTAINED'
  changed.decision_contract.render_evidence = 'UNAVAILABLE'
  changed.decision_contract.bends.forEach((row) => { row.state = 'UNKNOWN'; row.observation_detail = 'UNOBSERVED' })
  changed.decision_contract.observed_counts = { formed: 0, unformed: 0, unknown: 3, total: 3, partially_observed: 0 }
  const result = presentBendDecision(changed)
  assert.equal(result.available, true)
  assert.equal(result.registration, 'ABSTAINED')
  assert.equal(result.renderEvidence, 'UNAVAILABLE')
  assert.equal(result.counts.unknown, 3)
})
check('duplicate raw identifiers cannot inflate counts', () => {
  const changed = clone(report); changed.matches.push(changed.matches[0])
  assert.equal(presentBendDecision(changed).available, false)
})
check('countable flags exclude process features', () => {
  assert.equal(presentBendDecision(report).counts.total, 3)
  const nested = clone(report)
  delete nested.matches[3].countable_in_regression
  nested.matches[3].cad_bend = { countable_in_regression: false }
  assert.equal(presentBendDecision(nested).available, true)
  nested.matches[3].countable_in_regression = null
  assert.equal(presentBendDecision(nested).available, true)
})
check('page, annotations and decision contract share explicit countability', () => {
  for (const [match, expected] of [
    [{ countable_in_regression: null, cad_bend: { countable_in_regression: false } }, false],
    [{ feature_type: 'ROLLED_SECTION', countable_in_regression: true }, true],
    [{ feature_type: 'ROLLED_SECTION', countable_in_regression: false }, false],
    [{ feature_type: 'ROLLED_SECTION' }, true],
    [{ countable_in_regression: true, cad_bend: { countable_in_regression: false } }, true],
  ]) {
    assert.equal(isDecisionCountableBend(match), expected)
    assert.equal(annotationSandbox.exports.isCountableBend(match), expected)
  }
})
check('malformed effective countability cannot create positive evidence', () => {
  for (const token of [0, 1, 'false', 'true', {}, []]) {
    const changed = clone(report); changed.matches[0].countable_in_regression = token
    assert.equal(presentBendDecision(changed).available, false)
    changed.matches[0].countable_in_regression = null
    changed.matches[0].cad_bend = { countable_in_regression: token }
    assert.equal(presentBendDecision(changed).available, false)
  }
})
check('malformed or contradictory registration source cannot be hidden by a contract', () => {
  for (const alignment of [[], 'bad', 0, { alignment_abstained: 1 }, { alignment_abstained: 'true' }, { alignment_abstained: true }]) {
    assert.equal(presentBendDecision({ ...report, alignment }).available, false)
  }
  assert.equal(presentBendDecision({ ...report, alignment: { alignment_abstained: false } }).available, true)
})
check('row permutation does not change evidence counts', () => {
  const changed = clone(report); changed.matches.reverse(); changed.decision_contract.bends.reverse()
  assert.equal(JSON.stringify(presentBendDecision(changed).counts), JSON.stringify(presentBendDecision(report).counts))
})
check('labels distinguish partial visibility from physical formation', () => {
  const matches = presentBendDecision(report).matches
  assert.equal(decisionLabel(matches[0]), 'Observed formed · unverified')
  assert.equal(decisionLabel(matches[1]), 'Observed unformed · review')
  assert.equal(decisionLabel(matches[2]), 'Partly observed · unknown')
  assert.equal(decisionLabel({ decision_state: 'UNKNOWN', observability_detail_state: 'UNOBSERVED' }), 'Not observed · unknown')
  assert.equal(decisionLabel({ decision_state: 'UNKNOWN', observability_detail_state: 'UNKNOWN' }), 'State unknown · review')
  assert.equal(decisionLabel({ decision_state: 'UNKNOWN', observability_detail_state: 'OBSERVED_FORMED' }), 'State unknown · review')
})
check('presentation is idempotent and never mutates source measurements', () => {
  const first = presentBendDecision(report)
  const second = presentBendDecision({ ...report, matches: first.matches })
  assert.equal(JSON.stringify(first), JSON.stringify(second))
})
check('invalid count estimates never become display evidence', () => {
  for (const estimate of [-1, NaN, Infinity, 2.4, '4', null]) {
    const changed = clone(report); changed.summary.structured_completed_bends = estimate
    const result = presentBendDecision(changed)
    assert.equal(result.modelEstimate, null)
    assert.equal(result.counts.formed, 1)
  }
})
console.log(`${checks} synthetic decision-contract checks passed`)
