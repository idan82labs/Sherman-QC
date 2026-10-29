# Engine-first roadmap checkpoint — 2026-09-07

This is an implementation and synthetic-verification receipt, not physical
qualification, a release authorization or a replacement frozen protocol. The
user reaffirmed **engine first**, with safety-critical product work in parallel.
The attached long-running goal remains paused.

Subsequent same-day engine checkpoint:
[position-only candidate birth and association](2026-09-07_position_region_discovery.md)
now generates the three region inputs within a bounded, explicitly conditional
search. The verification receipts below describe the earlier implementation.

## Engine progress

The [position-derived axis method](2026-09-07_position_axis_hypothesis.md) removes
the need for a supplied exact direction, conditional on two explicitly supplied
plane neighborhoods. It uses positions only, all-point plane witnesses, SVD
normal cones and cross-product direction bounds. Parallel, thin/rank-deficient,
missing and unresolved neighborhoods abstain. Neighborhood discovery, shared
physical surface identity and calibrated uncertainty remain unsolved.

The centered-error proof does not assume independent or zero-mean errors. The
cross-product bound handles independent normal sign gauges and does not omit a
quadratic term. An independent Astra review agreed with these exact-arithmetic
bounds; the numerical safety allowances remain engineering guards, not interval
arithmetic certificates.

The normal/direction sweep covered **10,000 invented cases**, 50 positions each,
with varied scales, plane separation, rigid orientation and bounded perturbations:
5,873 bounded direction hypotheses, 4,127 abstentions, and 6,472 available axis
bounds all containing known truth. Agent receipt: 1.61671 seconds and 124,141,568
bytes peak process RSS. This is neither a native pipeline benchmark nor a
physical generalization result. Root independently ran the named analytic,
fixed-axis radius and position-axis suites: **269 tests passed in 0.29 seconds**.

The new [plane-to-radius bridge](2026-09-07_plane_conditioned_radius_bridge.md)
then carries this direction uncertainty into the radius diagnostic. For any
compatible true axis, one common rigid rotation of all true cylinder samples
preserves their radius and bounds their displacement by
`epsilon + 2 sin(delta/2)(observed_footprint_radius + epsilon)`. The resulting
radius interval is a necessary outer constraint under the stipulated joint
model. A fitted circle in that relaxed problem does **not** establish original
joint compatibility, a physical axis location or continuous material support.

An independently constructed 15-position counterexample makes the consequence
concrete: a 0.01-radian axis tilt projects exact radius-2 cylinder samples onto
a perfect radius-4 circle. Naive substitution falsely excludes radius 2 with a
narrow interval around 4. The bridge retains 2 and abstains. The independent
regression also uses truly tangent source planes, and separately challenges
near-tight correlated normal errors and false observed axial extent.

The bridge's expanded 1,000-case synthetic sweep produced 290 conditional radius
hypotheses and 710 abstentions. All radius constraints retained the constructed
true radius; all available rotation and axial-coordinate bounds contained their
constructed truth. Agent receipt: 0.59817 seconds, 52,576,256 bytes peak process
RSS. Root's final combined analytic/radius/axis/bridge/independent-math suite
passed **332 tests in 0.44 seconds**. This is mathematical regression evidence,
not detection accuracy or physical calibration.

The machine reports 18 logical CPUs and 137,438,953,472 bytes RAM (128 GiB).
Independent source/synthetic work was parallelized. Frozen per-job resource and
custody limits were not increased; none of these small vectorized jobs needed
the full memory budget.

## Production promotion boundary

The old aggregate gate permitted relaxed thresholds, unrelated equal-size CAD/PLY
cohorts, missing family coverage and a scalar repeat rate without ten repeats.
Separately, the API/worker accepted promotion booleans plus three hash-shaped
strings, even with active blockers. Those declarations were not verification.

The new pure stdlib
[`check_enterprise_promotion_gates_v2.py`](../../scripts/check_enterprise_promotion_gates_v2.py)
requires fixed thresholds, unique declared families/scan IDs, paired per-scan
results, complete operational denominators, candidate/protocol/evaluator/scope
bindings and at least ten whole-cohort semantic replays. Count, range, instance
precision, coverage and latency denominators are explicit. Nonfinite, malformed,
duplicate and unrepresentable declarations fail closed. The positive test uses
150 invented scans / 75 parts / five families / 300 lane jobs; it is not evidence
that a product candidate meets any threshold.

The v2 report is **evidence-contract eligibility only**, always non-authorizing.
Hashes bind supplied declarations; no referenced bytes, deployed candidate,
untouched truth or review authority were verified. The prospective instance
precision definition requires review before any frozen physical evaluation.
See the [updated promotion policy](../release/ENTERPRISE_BEND_PROMOTION_GATES.md).
Historical v1 outputs remain historical diagnostics and now explicitly state
that they cannot authorize production.

The API and isolated worker share
[`promotion_readiness.py`](../../domains/bend/services/promotion_readiness.py).
Legacy status remains a progress display, not a trusted release path. Missing,
malformed and self-declared verified/promoted/shadow states remain research-only;
neither the API nor worker grants production permission from the status file.
This does not implement the future candidate-bound evidence-byte/authority
verifier or a floor-owner release decision. It closes an unsafe shortcut.

## Operator access and artifact safety

Active bend routes require an authenticated active operator/admin. New jobs have
server-assigned numeric owner identity. Results, labels, artifacts, HEAD and
deletion are owner-bound; admins do not gain cross-owner access. Ownerless
historical jobs fail closed instead of being silently adopted. Completed-job
metadata preserves ownership across reloads. Full UUIDs and exclusive upload
directory reservation prevent collision-based artifact reuse. Active jobs cannot
be deleted underneath their worker.

The shared legacy jobs/reports routes are guarded too: merely protecting the bend
router left earlier mesh routes and generic report/upload downloads exposed.
Tests assemble the actual route definitions in both orders with synthetic storage
and identity lookup. They execute the source-extracted real authentication
dependency, but **not** real token crypto, accounts, database or native runtime.

Independent review found two additional namespace leaks, now fixed: temporary
CAD extraction files under `temp_*` are never served through shared aliases, and
the public general `/api/jobs` list excludes literal case-insensitive `bend_`
IDs in SQL **before both counting and pagination**, including part/status filters.
Default database callers preserve their prior behavior. SQLite/PostgreSQL list
and count predicates are aligned; the previously missing PostgreSQL `count_jobs`
method was added. SQLite query tests use an invented in-memory database;
PostgreSQL tests translate placeholders through a stub, not a live driver/server.

Remaining access limits: real JWT/account behavior and full native application
integration are untested; filesystem metadata/symlinks are trusted. Running-job
ownership is process-local until completion, so restart/multiworker operation may
deny legitimate access. A durable ownership/admission ledger remains necessary.
The public general `/api/stats` aggregate endpoint is unchanged and out of scope.

Artifacts now use same-origin authenticated fetch, including the Three.js binary
path; tokens do not enter URLs and redirects are refused. Images have loading,
retry and unavailable states; PDFs use authenticated blob downloads. The secondary
Bender View now uses the versioned observation projection and removes legacy
PASS/FAIL verdicts and machine-adjustment advice. This was safety integration,
not a visual redesign. The UI/UX skill guided local error/retry feedback and
cleanup. Remote HDR lighting was removed, allowing local rendering without that
network dependency. URL changes remount geometry state and cancel stale fetches.

## Verification boundary and receipts

Root verification, dependency-isolated and using only named source/new temporary
synthetic files:

- Final product-support suite: **618 passed**, plus 112 subtests in 11.77 seconds
  (one existing Starlette/httpx deprecation warning). Covers ownership/shared
  aliases, aggregate namespace isolation, API/worker readiness, both promotion
  checkers, runtime observation/decision boundaries, match/report projection,
  stubbed PDF projection and in-memory UI contracts. No live service was started.
- Final five-file geometry suite: **332 passed** in 0.44 seconds, including
  [five independently authored mathematical challenges](../../tests/test_axis_radius_independent_bounds.py).
- Client VM/React SSR/source-extracted loader suite: **69 passed**. Existing
  observation presentation suite: **17 passed**.
- Full TypeScript/Vite build and full frontend lint passed. Existing large-chunk
  and stale Browserslist warnings remain; no dependencies were changed.
- Named modified-source whitespace checks passed.

Compiled UI was checked through the browser skill against the new disposable
[`serve_bend_floor_synthetic_ui.mjs`](../../scripts/serve_bend_floor_synthetic_ui.mjs)
loopback harness. Its APIs, operator identity, four-point PLY and image bytes are
invented; it has no real backend or proxy and serves only freshly built frontend
assets. CSP restricts requests to that origin (plus local image blobs).

The check exercised synthetic sign-in, the main report, authenticated scan and
reference parsing, image display, an intentionally unavailable PDF, secondary
report warning, one failed image request and successful retry. Browser inspection
confirmed 1/3 observed formed despite raw legacy totals of 999, no raw machine
advice, and local blob-backed images. The secondary view was visually inspected
at 390px with no horizontal overflow; desktop was 1728px with no overflow.
The main viewer created a 1720×1120 canvas; no captured warning/error logs were
returned in the final check. Harness counters recorded no rejected authentication
requests and no unexpected API paths. No actual PDF layout was tested. The
temporary viewport was reset and the tab and harness were closed afterward.

The earlier standalone compiled-UI test's expectations were updated for local
lighting but that standalone browser runner was not executed this turn; current
visual evidence comes from the browser-skill session above.

## Remaining release blockers

Do not infer that the production floor is qualified from these improvements.
The next engine work is region discovery/association and joint, position-derived
finite-bend measurement without hidden oracle channels. Research methods enter
the active detector only after their candidate-bound evidence supports that use.

The roadmap still requires capture/calibration/frame/stage provenance, native
candidate verification, durable queue/admission/process-tree cancellation/restart
recovery, authenticated evidence-byte/reviewer verification, prospective physical
evaluation and a scoped operator-reviewed shadow release with rollback/monitoring.
Public general-system aggregates and trusted-filesystem assumptions are not a
complete application security audit. C15's three existing source-freeze failures
remain; EXP35 remains terminal. No real inputs, held-outs, incoming-email parts,
control/issuance/one-shot records, saved scientific outputs or real services were
read or executed. Existing unrelated worktree changes were preserved.
