# V9 completion lane — final stable-checkpoint closure audit

Date: 2026-09-13. Reviewer: Codex Astra subagent `/root/v8_finite_start_audit`.

## Disposition

**The identified completion-lane source defects are closed at the checkpoint
hashes below. The claimed connected completion-only test closure is not yet
established by the submitted test source.** No new P1 inference defect was
found in this bounded review. One P2 validation requirement remains: exercise
an honestly completion-only candidate through the full V16 registration-binding
path and into measurement dispatch, rather than calling only the origin helper.
The author's stable response overstates that particular test's scope.

This is the requested one-pass closure review, not a physical PRE, freeze,
execution authorization, score, or production qualification. The reviewer read
AGENTS, the latest status checkpoint, the complete named review response, and
the relevant source/test seams. No imports, tests, geometry jobs, physical/raw
inputs, Gold, attempt records, or private caches were accessed. The author
remains the sole engine/test editor. Only this new audit document was written;
all prior reports remain unchanged.

The author's response reports 826 passing tests at this checkpoint. That is
**author-reported execution evidence**, not an independent execution result
from this audit. Test source can establish what a test checks, but not that
its claimed run occurred or passed. Root owns execution-evidence acceptance.

## Hash binding

All named source/test hashes matched the stable response when inspected:

| File | SHA-256 |
| --- | --- |
| `docs/research/2026-09-13_fable51_v9_review_response.md` | `31def50d1d4795c22f8c2c79b8899eafe445c388ecfd7041b1c4697da0985de4` |
| `backend/cad_guided_plane_initialization_v15.py` | `505f313dbf3e0487bf03f3b92ab1d42d90f61d5df3123c34c9a91677d132647e` |
| `backend/cad_guided_mesh_offline_v16.py` | `5b48ad81c2d1796529c6c43cd7f27b48f73413b50f3adb74d9e6cf9608519012` |
| `backend/cad_guided_report_envelope_v7.py` | `b7abe653c6f7b9832c61687ec7f1ed68aa44361b8ef467b8cae84bf837bef075` |
| `tests/test_cad_guided_plane_initialization_v15_completion.py` | `9a0688b2e8cfe7e96fdca5a38e7a07b05735a9641d07638a97038637a2c403fc` |
| `tests/test_cad_guided_plane_initialization_v15_registration.py` | `6b36dba73a145435470f6854679805e5ef7070a2ff9fb61eda4cb12d8efb491c` |
| `tests/test_cad_guided_completion_v1_engine_integration_v1.py` | `0927168c56caf7764aed5f22d109c45e951a3482eef56a6af69bb4b13e08cee4` |
| `tests/test_cad_guided_report_envelope_v7.py` | `b1165b8a57652986ec88b0cd00df6f50a616425f11499a660c3bab2252bac887` |

The prior bounded delta report remains
`2026-09-13_gold11_v9_completion_delta_codex_audit.md`, SHA-256
`9c58a6d1f15cf7e2e9b7e8b2153bb576764cc1b7d933e9cece717e4449dea2c2`.
This report evaluates its named remaining findings, without reopening generic
geometry research or claiming to cover the separate kernel/material/runner
audits.

## Closure matrix

| Prior issue | Source disposition | Test-source disposition |
| --- | --- | --- |
| D1: lane cap precedes last mutation or skips empty/disabled lane | Closed for explicitly declared non-fatal terminal paths and normalized-size convention | Exact boundary, post-schedule inclusion and disabled path covered; purported zero-child cap case actually schedules one child |
| D2: replacement origin not bound to unique rejected parent slot | Closed | Honest helper-origin binding and ten targeted stale-field/pose refusals covered; not the full public binder |
| D3: expected alternative shift ends in `or True` | Closed | Signed origin/alternative shifts are asserted without escape hatch |
| D4: no connected completion-only or late-fatal evidence | Source has full binder path and whole-failure invalidation | Late fatal after naturally passing completion covered; connected completion-only public binding/dispatch remains absent |
| Envelope malformed top-level evidence implies false zero completion count | Closed | Existing malformed completion-container tests remain; malformed top-level `evidence` branch has no dedicated test |

## D1 — Final lane cap and fatal scope

`backend/cad_guided_plane_initialization_v15.py:912` now centralizes cap
enforcement in `_cap_completion_lane`. It is called on the disabled path at
`:718`, the no-child path at `:854`, after full child evaluation and COMPLETE
status at `:909`, and after the final `refinement_schedule` publication at
`:1124`. On a cap error it raises into the existing whole-initialization
failure handling; no cap is increased and no passing start is released from
that failure.

The size convention is now explicit at `:709`: compact JSON bytes with
`ledger_bytes` replaced by null, checked after the last mutation on every
**non-fatal** terminal path. `_cap_completion_lane` implements exactly that
normalization at `:918` through `:923`. Thus the actual serialized lane can
differ by the numeric size field's digit count; `ledger_bytes` is not claimed
to be the literal final wire size. This declared accounting convention
addresses the earlier ambiguity and is distinct from the outer report's
actual serialized-byte limit.

After a fatal failure, the lane receives invalidation fields at `:1185`
through `:1191` and is not re-capped as a successful lane. Its normalized
`ledger_bytes` can therefore be absent or describe the earlier non-fatal
snapshot. This is within the newly declared non-fatal scope, not an
all-terminal-path byte guarantee. The whole result still passes through the
outer bounded report at `:1198`, preserving its existing refusal/accounting
behavior. A future summary should not shorten the scope to an unqualified
"all terminal paths" claim.

The test at
`tests/test_cad_guided_plane_initialization_v15_completion.py:379` checks
post-schedule normalized-size equality. It derives a same-digit fixed-point
cap from the lane's embedded budget, checks exact acceptance and one-byte-less
refusal, and exercises the disabled path. However, the comment at `:410` says
"zero selected children" while the call at `:414` through `:416` supplies a
known compatible parent and allows one child. This is a one-child tiny-cap
refusal, not an empty-lane branch test. The source fix for an empty lane is
present and straightforward; the test description/fixture should be corrected
without expanding the experiment. A truly empty candidate set with a retained
parent can use the existing declared unresolved-pair fixture, and should
assert that the retained child list is actually empty before attributing
coverage to this branch.

## D2 — Exact parent-to-child replacement mapping

`backend/cad_guided_mesh_offline_v16.py:116` derives the non-PASSED parent
positions. It requires a three-support rejected triplet and exactly one such
position (`:117` through `:123`), then specifically requires that position to
be REJECTED (`:130`). The old "replaced ID occurs anywhere in the parent"
condition has been removed.

The new block at `:130` through `:140` binds the replaced CAD ID, retained scan
plane ID, replacement CAD ID and sign, complete sign vector, and both
unchanged passing correspondences/signs. The completed face cannot reuse a
passing parent face. The lane's parent record must agree on failed slot,
failed face/scan, axis, and child membership at `:141` through `:147`.
Exact parent-plus-shift translation and complete passing-origin membership
checks remain at `:148` through `:154`. Child supports still have to agree
with child correspondences and all pass (`:102` through `:112`). This closes
the specific stale replacement-history publications identified in D2.

The test at
`tests/test_cad_guided_plane_initialization_v15_completion.py:423` obtains
actual completion children from the finite completion helper, then binds the
honest mapping and mutates replaced/completed IDs, sign, passing parent
correspondence/sign, parent slot/index/axis, support status, and axial shift
at `:449` through `:461`. These are meaningful helper-contract negative
controls, not evidence of physical wrong geometry.

The producer still retains child origins before fallible finite work and
re-evaluates all three supports. Nothing in this delta reintroduces inherited
PASSED flags or CAD-derived measured answers.

## D3 — Signed-shift assertion

The prior `or True` has been removed. The recovery test at
`tests/test_cad_guided_plane_initialization_v15_completion.py:180` through
`:194` derives the origin's correction as `-shift * (axis dot u)` and an
alternative face's correction as that value plus
`(z_face-z_origin) * u_z`. The test first establishes `|u_z| = 1` within its
declared numerical tolerance, so this formula agrees with displacement along
the fixture's plus/minus z axis. Both parent shift signs are parameterized at
`:161`. It separately asserts the alternative at z=10 passes and the z=30
shift loses formerly passing finite support (`:194` through `:198`).

This is now a signed pose-consistency check, not merely an absolute-value
comparison. It remains a limited orthogonal invented fixture and does not
claim calibrated physical accuracy or cover every canonical-normal boundary.

## D4 — Late fatal closed; connected completion-only validation still open

The new test at
`tests/test_cad_guided_plane_initialization_v15_completion.py:467` uses the
natural-parent geometry and the real initializer. It injects a late failure
only after at least one completion has passed, at the lane-cap/publication
seam. Assertions at `:479` through `:488` require no returned context, a
non-COMPLETE result, an unusable invalidated lane, retained PASSED child
evidence, all selected children counted as attempted, no scheduled starts,
and cleared start-group poses. This directly covers the earlier
fatal-after-passing-child concern at the initializer boundary. It is a
publication-seam injected failure, not a query/build exception test and not a
whole measurement-workflow execution.

The completion-only test does **not** call V16 `_registration_binding` or
`inspect_cad_guided_mesh_v16`. It calls only
`_validate_completion_origins` at `:439` and `:448`. The surrounding public
binding requirements at `backend/cad_guided_mesh_offline_v16.py:290` through
`:348`—candidate/selected/all-group agreement, scheduled disposition,
conditional context contract, both origin families, and final selected
pose—are not traversed by that test.

The separate integration fixture at
`tests/test_cad_guided_completion_v1_engine_integration_v1.py:478` creates
three groups with ordinary `hypothesis_indices=[index]`; its live public
binding wrapper and measurement-dispatch test still exercise those main-only
groups. The unmocked registration suite uses the two-sheet fixture with
`max_bank_planes=2` at
`tests/test_cad_guided_plane_initialization_v15_registration.py:50` through
`:80`, so it does not supply a three-support completion-only origin either.

Therefore response-map rows 14 and 18 do not establish the stated connected
completion-only closure. This matters because the original P1 was precisely
a seam mismatch where a valid helper-created completion could be rejected by
the workflow's ordinary-origin requirements. The source now appears to fix
that mismatch, but a helper-only test cannot guard against its recurrence.

Smallest remaining action: add one bounded invented-input integration case
whose selected candidate has no ordinary origins and valid retained
completion/parent ledgers, call the **actual public V16 binder** and require
measurement dispatch to be reached. Native loading/measurement output may
remain explicitly substituted for this wiring check; the binder must not be
substituted. Add selected-group completion-index mismatch and stale-origin
refusal checks through that same public path. No physical data, benchmark,
new geometry method, or successful measurement score is needed to close this
test requirement.

The comment at completion-test `:462` is also incorrect: mixed
ordinary/completion origins are not categorically impossible. V15's exact-pose
lookup can find an existing ordinary group and append `completion_indices` at
`backend/cad_guided_plane_initialization_v15.py:899` through `:903`. This is a
legitimate producer branch if the pose bytes coincide; it was not observed
in a run by this reviewer. V16 validates both origin families for such a
group (`:313` through `:332`). The test currently asserts only that its own
constructed group has no ordinary origins; it does not test a forged main
origin or a genuine mixed group. Correct the comment and, within the same
small binding matrix, retain a mixed-origin case rather than treating it as
invalid by definition.

## Envelope malformed accounting

`backend/cad_guided_report_envelope_v7.py:181` now adds an unknown completion
count when the top-level `evidence` container is malformed. The existing
merge rule preserves a direct known count as a lower bound if such a count
also exists. The docstring at `:6` correctly states that only canonical V7
refusals are exempt from the detailed cap. Both earlier source notes are
closed.

The eight-test envelope file is unchanged from the previous checkpoint. It
covers malformed completion-lane containers, not malformed top-level
`evidence`. Add an explicit example of each of the two expected outcomes
(unknown with no direct list; known lower bound with a direct list), or label
this branch as source-reviewed only. The lack of those two assertions does
not negate the inspected one-line source correction.

## Bounded handoff

All remaining changes requested here concern already-identified closure
claims and test paths. No new research phase is proposed. Main-first
refinement scheduling, non-exhaustive |lambda| ranking, separate completion
budgets, provisional units, all-three finite support, and honest fatal
abstention remain the reviewed design. Root is separately reviewing the
runner's active/inactive budget declaration and total experiment limits;
this report does not approve that freeze wiring.

Accept the source-level closures above, correct the exact test-scope claims,
and complete the small public-binding integration check before claiming full
completion-lane closure. A later hash or test result is not covered by this
document automatically. No historical F1 improvement, new physical score,
unseen-family generalization, manufacturing truth, or production release is
claimed.
