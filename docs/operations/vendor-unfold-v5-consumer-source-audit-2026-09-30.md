# V5 maintained-consumer source audit — 2026-09-30

## Scope and decision

Independent source/synthetic audit of the maintained vendor browser admission,
native STEP upload V2 versus V1, and maintained upload/unfold runtime selectors.
No physical CAD, raw scans, Gold, corpus run, live database, native geometry job,
browser session, deployment, or admission-lock mutation was used. This is not a
visual-integrity, manufacturing, or physical-metrology qualification.

**Final maintained-consumer source decision: GREEN at the recorded identities.**
The post-repair recheck below closes the initial source-pin and upload-route
harness failures. This reviewer greenlights the source/consumer boundary for a
separately audited frozen native experiment, not the experiment's geometry,
physical outcome, deployment, or manufacturing release. Initial failures remain
retained below; passing mocked transport tests alone did not establish pins.

## Post-repair independent recheck

After the parent declared source stabilization, this reviewer independently:

- Reran the combined upload V2, V5 preview, transport V4, maintained publication,
  and API contract suites: **108 passed in 0.48 seconds**.
- Read and ran the new durable
  `frontend/react/scripts/check-vendor-unfold-v5-contract.mjs`: **1 synthetic
  V5 accepted / 13 malformed refused**, including full oracle-bearing receipt,
  strip-width/work corruption, source/member partition, overlay labels/parity,
  winding, typed versions, and missing/overclaimed oracle cases.
- Reran incumbent browser checks: V3 **13/128**, V4 **10/125**, unchanged.
- Reran the independent temporary cross-language probe: **2 V5 accepted / 30
  malformed refused**, UUID/list-order independence and no mutation.
- Reran actual read-only pin closure: V2 upload **935 pins**, V5 transport
  **934 pins**, both passed. No native process was launched by these checks.
- Rechecked every recorded consumer identity: all matched. Reviewed repaired
  route fixtures now explicitly cover default V1 and development V2 with mocked
  native producers, including previous-publication retention; added V5 API
  cases cover absence, worker crash, candidate-generation change, and unchanged
  candidate/draft snapshots.

The parent separately reported passing TypeScript/build checks and a bounded
authored STEP → upload V2 → V5 oracle smoke test. Those are parent-run evidence,
not native/visual verification performed by this source-only reviewer.

## Independent evidence

1. Existing browser synthetic boundary command:

   `cd frontend/react && node scripts/check-vendor-unfold-contract.mjs`

   Passed: 13 V3 accepts / 128 malformed refusals; 10 V4 accepts / 125 malformed
   refusals; 10 standalone add-on checks. Old V3/V4 marker support, Float32
   noncollapse, triangle identity, IDs/order/groups/colors, and no source
   mutation checks remained passing.

2. Independent cross-language V5 probe:

   `node /tmp/vendor-v5-consumer-independent-audit.mjs`

   The probe builds only the authored split-two-bend synthetic chart fixture,
   uses the independently assembled capture/oracle test helper, attaches the
   declared model receipt and current engine hash, transpiles the actual TS
   consumer and both imported diagnostics helpers, and passes the JSON directly
   between processes. It accepted the complete V5 receipt and a UUID/list-order
   permutation; refused 30 independent corruptions; and verified unchanged
   input arrays. The malformed probes covered typed versions, cache profile,
   source hash syntax, native/chart namespace, source cardinality/bijection,
   member partition, source slices, selected roles, winding declarations/index
   swaps, complete overlay geometry and exact feature ownership, body parity,
   and oracle status/model/continuous-area declarations. No physical inputs.

   Temporary probe SHA-256:
   `8c81de0236861f1884ffd9f9828f4b6c6fa67f8ecba401994617532cb9053577`.
   This temporary probe is supplemental audit evidence, not a durable product
   regression command; the V5 cases should be persisted in the repository
   harness before treating that harness as complete V5 coverage.

3. Focused source-only backend command:

   `.venv/bin/python -m pytest -q --noconftest tests/test_vendor_native_step_upload_v2.py tests/test_vendor_unfold_preview_v5.py tests/test_vendor_unfold_worker_v4.py`

   Passed: **89 tests**. Transport tests intentionally mock source hashing and
   native subprocesses, so this does not certify current actual dependency pins.

4. Maintained route boundary command:

   `.venv/bin/python -m pytest -q --noconftest tests/test_vendor_cad_upload_publication.py tests/test_vendor_unfold_api_contract.py`

   Initial result: **16 passed, 1 failed**. Failure was
   `test_step_upload_uses_native_source_and_retains_previous_on_native_refusal`:
   extracted route scope lacked `os`, causing `CAD_ANALYSIS_FAILED_NOT_PUBLISHED`
   with `name 'os' is not defined`. The production module already imports `os`;
   this is a regression-harness defect, not evidence of that production NameError.

5. Real read-only source-pin check:

   `.venv/bin/python -c "from backend.vendor_native_step_upload_v2 import _source_pins; print(len(_source_pins()))"`

   Initial result: fail-closed `UNFOLD_V4_SOURCE_NOT_AUDITED`. Source closure was
   changing concurrently. Refresh pins only after technical review of final
   bytes, then require this check to pass; never disable it to obtain a run.

6. Additional AST-only V5 API selector probes, using the source test helper
   with `runtime='v4-development'`: authored unsupported geometry returned
   `SYNTHETIC_UNSUPPORTED`; missing CAD returned `CAD_SOURCE_UNAVAILABLE`;
   concurrent candidate-generation change returned
   `CAD_CANDIDATES_CHANGED_DURING_PREVIEW`; and an authored worker exception
   returned `UNFOLD_WORKER_UNAVAILABLE`. These independent executions passed;
   they do not replace durable test functions covering that runtime.

## Mechanism conclusions

- Browser V5 admission requires mechanism integer 5 / transport integer 4,
  controlled profile, native/chart namespace, complete chart/member ownership,
  retained source triangle partition, positive-Z index winding, local-strip
  proof declarations, exact ordered child-chart overlay parity, and the
  declared discrete native metric-chart oracle receipt. It does not read CAD
  or independently prove native geometry. Backend pinned producer and oracle
  remain the evidence boundary; source hash syntax in the browser is not a
  cryptographic proof of execution.
- V3 and V4 continue on their distinct prior admission branches. V5 is not
  admitted by weakening or relabeling a V4 browser receipt. Existing V3/V4
  boundary regressions pass.
- V2 upload diff preserves V1 candidate/frame/mesh checks, byte snapshot,
  parent result validation, source-change checks, supervision, and budgets.
  The intentional changes are V5 reader/dependency worker, separately named
  profile/nonce/subprocess/temp namespace, and the V5 flat-preview declaration.
  The new extractor's mathematics was outside this independent audit because
  this reviewer authored that separately versioned file.
- `SHERMAN_VENDOR_CAD_RUNTIME=v2-development` selects V2; unset selects V1;
  unrecognized values fail the upload without publishing. The separate
  `SHERMAN_VENDOR_UNFOLD_RUNTIME=v4-development` selects V5 transport 4;
  earlier explicit runtime branches remain present. Both flags must be set
  deliberately for an end-to-end candidate experiment; neither implies
  manufacturing qualification. Default behavior is not silently promoted.
- Maintained upload publication still verifies exact CAD hash and folded mesh
  hash before committing the reserved version. Failure/supersession does not
  replace prior published CAD/candidates; generated artifacts have attempt-owned
  paths. Unavailable flat geometry must remain unavailable, not be displayed as
  a valid unfolded result.

## Closing checks and follow-up priorities

Closed before freeze: reviewed worker/helper pins pass the actual V2/V5
`_source_pins()` check; route scope is repaired; prior-publication preservation
cases pass under both upload selectors.

Closed regression coverage: default/development upload selectors and V5 unfold
failure paths now have durable source-only tests. P2 extra coverage: add a
maintained upload unknown-runtime test asserting no native producer invocation;
the current source fails closed before producer selection, and existing unfold
unknown-runtime regression remains passing.

Closed consumer coverage: the separately named durable V5 browser harness now
retains the complete synthetic oracle-bearing receipt and negative probes.
These are admissibility tests, not visual or oracle calibration. The temporary
independent audit adds the broader body-truncation and UUID permutation probes.

P2 after permitted native execution: visually check the maintained app's folded
and flat body, every full flange overlay and bend ID/group, touch selection,
rotation/zoom/pan, and truthful unavailable state. This source-only audit did
not perform that step.

## Recorded consumer identities

| File | SHA-256 at audit |
| --- | --- |
| `frontend/react/src/features/part-catalog/pages/vendorUnfoldPreview.ts` | `fc3f7b918d8d2011a9b272ac973a12976d6b11593a16bb2a2ab16eda42735403` |
| `frontend/react/src/features/part-catalog/pages/vendorUnfoldLocalStrip.ts` | `b2233ac66117c63fa916468825f2045f1af986b727aff802e3ce395decbc89ae` |
| `frontend/react/scripts/check-vendor-unfold-contract.mjs` | `7f60ed67fc905882116e98b8c0ddc83aaa11785f85d37197864aa401bff3cd64` |
| `backend/vendor_native_step_upload_v1.py` | `21e6f9cec6f855789c18ca033f637545cccbfb2ffc4fa121e1b4b55c6764fc0c` |
| `backend/vendor_native_step_upload_v2.py` | `ba40c980af5316f79158f878dec104afd103f7486c59b6721929bd90adaa8dfe` |
| `backend/_server_legacy_impl.py` | `736d0aa7f4d762f5b1d191ee89a15910a890bdac0def40a125a576d7a6033403` |
| `apps/api/services/vendor_setup_service.py` | `29dd07340c38147b4566bd33ae70c4d6fa1323c5e3a1e5796358d04cb7ceccbb` |
| `frontend/react/scripts/check-vendor-unfold-v5-contract.mjs` | `0a75b3099122b54da2d279018c2e5eddd8590b58eb0abc6e78c6b7eb8935bb60` |
| `tests/test_vendor_cad_upload_publication.py` | `a69747a9de771831cfac837d7d6ff7a79a9af8bc217dac5f47312a2b1dda5399` |
| `tests/test_vendor_unfold_api_contract.py` | `bbbd9a6b190b132f12128ec0c4bee01b1b088fcbecfeee065758f59b39061683` |
| `backend/vendor_unfold_worker_v4.py` | `e005857d485096bb24c9fdb458a293a108d9ab871db112fcb2dca4f11be8d686` |
| `backend/vendor_unfold_contract_v4.py` | `163f971c215084653ec1352e6f7614f1566982d368d5a84f7ff3268b08050642` |
| `backend/vendor_unfold_common_contract_v5.py` | `ffa6a42bd04f292ab38b34f43d3990285d54b6bfdec1ebae36cc8008d23658a0` |

Any relevant subsequent change invalidates the corresponding hash-bound result
until focused re-audit. An upstream worker/contract mathematical freeze is
separate from the consumer semantic review. The final worker/contract hashes
above record only the independently observed stabilized source-pin closure.
