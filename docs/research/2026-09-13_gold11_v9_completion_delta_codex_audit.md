# V9 completion lane — independent source-only delta audit

Date: 2026-09-13. Reviewer: Codex Astra subagent `/root/v8_finite_start_audit`.

## Scope and disposition

This is a bounded source review of the fixes following
`2026-09-13_gold11_v9_completion_source_codex_audit.md`. It is not a PRE,
execution authorization, test result, physical-input inspection, or qualification
of V9. Fable remains the sole engine/test editor. The reviewer performed no
imports, tests, geometry jobs, or physical/Gold/attempt/cache access; only named
source/test/released-document reads and source hashing. Only this new report was
written. Previous audit reports remain unchanged.

**The three original P1 mechanisms have source-level fixes in the revisions
below:** completion-only origins now have a consumer path; selected completion
children are retained before fallible finite work; report-envelope V7 accounts
for retained completion children. The replacement-face normal-sign issue is
also addressed. This is materially better source, but not complete closure:
the final completion-ledger byte cap is not enforced as declared, the new
origin validator does not fully bind the replacement mapping, and the reviewed
test source lacks a connected completion-only origin case and contains one
vacuous mathematical assertion. These are bounded follow-up items, not a reason
to abandon the completion hypothesis or widen the physical run.

## Exact revisions

Hashes bind this review, not any subsequent Fable edit:

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_plane_initialization_v15.py` | `3bd4840ef68940822ccf796c9a8bedfd2da4131bd95cc37ff25da7b1a909eb30` |
| `backend/cad_guided_report_envelope_v7.py` | `e75cf7bcce18434ed44b01fae1ce2537f2394089542416d8191324abc138b488` |
| `backend/cad_guided_mesh_offline_v16.py` | `7e8bf15defb0bb97188dab2de976406dad01141eae678e8439221c34cc8b761e` |
| `backend/cad_guided_mesh_offline_v8.py` (narrow predecessor array-binding seam) | `afdfc7f859eb5fdbf0fa7d796cc06a4180d40c02f6b0e590362f83e3f339e7ec` |
| `tests/test_cad_guided_plane_initialization_v15_completion.py` | `5ae79db310dffc1fbb6ed54ab1cd5b1f07ca53db36e03cd59eb48a335c723bb5` |
| `tests/test_cad_guided_report_envelope_v7.py` | `b1165b8a57652986ec88b0cd00df6f50a616425f11499a660c3bab2252bac887` |
| `tests/test_cad_guided_completion_v1_engine_integration_v1.py` (named fixture/binding seams) | `a969ec51f0e27547730c020e3f4c918e989f6cfb6536fe9d9d4015aa5b63d9d3` |

V16 was actively edited during this review. An earlier observed WIP revision
`3a2a80ee89a94b55abd7e6deb247ed0bc7da6b36274c87f9f7bc5037b052f4ae`
still contained the old nonempty-main-origin block. That observation is
superseded by the reviewed `7e8bf15d…` revision and is **not** a finding against
the latter. Likewise, the integration-test file changed from `8507e7fb…` to the
hash above while its frozen-copy comparison was updated. No test outcome is
inferred from either revision.

## Source-level closure of prior findings

### Completion-only groups can reach the current consumer

V15 still honestly creates a new completion-only group with
`hypothesis_indices=[]`, `lane=TRIPLET_COMPLETION`, and separate
`completion_indices` at `backend/cad_guided_plane_initialization_v15.py:888`.
It does not forge an ordinary triplet origin. The group is copied into the
refinement candidate at `:1427`, preserving the separate completion indices.

At `backend/cad_guided_mesh_offline_v16.py:282`, the current public binder allows
an empty ordinary-origin list only with the completion-only group lane and
completion origins. It compares the completion indices across candidate,
selected group, and all-passing group, and calls `_validate_completion_origins`
at `:290`. Ordinary origins remain subject to their existing complete-ledger
checks at `:291` through `:308`; a mixed ordinary/completion group is not allowed
to skip either origin family.

The preceding call at `:143` is deliberately the V8 array-only binder, via the
V12 `_previous` alias, not the V12 selected-candidate validator. Inspection of
`backend/cad_guided_mesh_offline_v8.py:64` through `:104` confirms that this seam
binds scan/CAD arrays and row selection and does not read a selected context
index. Its lack of a `context_index` argument is therefore **not** a defect.
V16 separately binds the context index at `:216` through `:232` and the selected
pose at `:310` through `:324`.

The remaining replacement-mapping validation gap below is distinct from the
old guaranteed refusal of every completion-only start. The guaranteed-refusal
mechanism is closed at this source revision; connected execution still needs
independent evidence.

### Selected children survive fallible work; fatal prefixes remain unusable

At `backend/cad_guided_plane_initialization_v15.py:819` through `:837`, every
selected child is appended to the public lane and its parent's index ledger
before support binding, face-index construction, screening, or querying.
Statuses mutate in place. The preflight row/call count is calculated before
`_bound_scan` at `:854` through `:864`, and distinct scan patches are bound once
at `:865` through `:872`. This addresses both the lost-selected-child failure
and repeated preflight binding of the same patch.

Finite evaluation still processes each of the three current supports, using
the child's translated pose (`:614` through `:681`). The previous two PASSED
flags are not copied as fresh support decisions. A far shift can consequently
invalidate formerly supported faces. Every selected child must be evaluated,
and the completed finite/box-excluded support denominator is checked at `:900`.

The outer initializer marks the lane unusable and passing children invalidated
after any fatal exception at `:1170` through `:1176`. All passing-group poses
are cleared and the context is discarded at `:1177` through `:1182`; the
registration wrapper also clears usable context publication after failure at
`:1470` through `:1482`. I found no favorable-prefix promotion in this seam.
Generation can still fail before a future child has been selected; the retained
ledger is attempted/selected work, not a claim to enumerate unseen future work.

### Envelope V7 retains known completion-discard counts

V15 now imports V7 at `backend/cad_guided_plane_initialization_v15.py:56` and
uses it at both initializer and registration publication boundaries.
`backend/cad_guided_report_envelope_v7.py:164` through `:205` inspects
`completion_lane.hypotheses` directly and under `evidence.initialization`.
Canonical V7 nested refusals preserve prior completion counts and first-refusal
evidence. The event schema and validation include the new count/status pair
at `:33` through `:40`, `:88`, and `:235` through `:242`.

The source tests cover both depths, canonical idempotence, malformed completion
containers, unchanged within-cap output, and forged completion counts in
`tests/test_cad_guided_report_envelope_v7.py:36` through `:80`. These are test
definitions, not executed results. The V6 canonical-envelope incompatibility
is explicitly tested rather than silently treated as V7 evidence. The known
direct/nested completion-list loss from the prior report is closed at source.

### Candidate-specific sign and axial math are consistent

`_candidate_sign` at `backend/cad_guided_plane_initialization_v15.py:589` chooses
the smaller atan2 angle for `+m` and `-m`. The new face's sign is written into
both the replacement correspondence and the complete sign vector at `:813`
through `:816`, instead of inheriting the old face's sign.

With world-frame scan witness `y=R*w+t`, unit candidate normal `m`, a point `a`
on its CAD plane, and unit passing-pair axis `u`, the code uses

`lambda = m dot (a-y) / (m dot u)`, then `t_new = t + lambda*u`.

The numerator and lambda have units mm; the denominator is dimensionless.
Flipping `m` cancels in the quotient; reversing `u` reverses lambda but not
the translated pose. The producer's row-vector point transformation
`points @ R.T + t_new` and column-vector witness `R @ w + t` agree. The source
records numerator, denominator, residual, candidate sign, and anchor at
`:823` through `:828`. The near-parallel passing pair and weak third-normal
projection are explicit unresolved cases (`:751` through `:758`, `:575`
through `:579`), not fabricated finite poses.

These are nominal plane-construction calculations under the declared mm model,
not propagated scanner/calibration uncertainty. In particular, the fixed
1e-9 numerator check labels a near-null branch as unconstrained; it does not
prove a mathematical exact nullspace. No candidate is admitted on that label.
All actual support remains finite and scan-derived. The physically relevant
sign test should use nearby normals straddling the producer's canonical-sign
boundary, not merely arbitrarily flipping an internal dictionary.

## Remaining actionable findings

### P2-D1 — Completion report cap is checked before final publication, and skipped on empty lanes

At `backend/cad_guided_plane_initialization_v15.py:904` through `:907`, encoded
lane size is checked while `ledger_bytes` is `None` and status is INCOMPLETE.
The code then changes those fields (`:908` through `:909`). Later `_prefilter`
adds `refinement_schedule` at `:1106` through `:1109`. No completion-lane byte
check follows those mutations. The no-selected-child branch at `:849` through
`:851` returns without checking the lane at all, even if retained parent and
omission records are nonempty. The disabled branch similarly returns early.

Therefore `ledger_bytes` is not the final published lane's size, and the final
lane can exceed `completion_budgets.max_report_bytes` while returning COMPLETE.
The separate outer initialization/report envelope still imposes its own cap;
this is not evidence that process memory is unbounded or that physical output
has already been lost. It is a reproducibility/cap-contract defect in the new
lane.

Smallest fix: apply the lane cap after its last mutation on every terminal
path, explicitly define whether the size field includes itself, and handle a
violation through the existing whole-initialization refusal. Add exact-boundary
tests, including zero selected children, a populated parent ledger, and the
post-schedule field. Do not raise an already frozen cap to accommodate output.

### P2-D2 — Origin binder does not fully validate which correspondence was replaced

`backend/cad_guided_mesh_offline_v16.py:115` through `:122` requires the reported
replaced CAD ID to occur somewhere in the parent's supports, not specifically
at its only rejected slot. It also does not require the completed CAD ID to
equal the new CAD ID at that child's replacement slot. The two passing
correspondences and their signs are not compared against the corresponding
parent slots. Child supports agree with child correspondences at `:104`, but
that is a separate relationship.

Consequently a stale replacement ID naming either passing parent face, or a
different unused completed-face label, can satisfy the current validator while
the child pose/support rows remain unchanged. Such a publication is not bound
to the claimed parent-to-child replacement history. The producer currently
constructs the intended mapping at V15 `:812` through `:828`; this review does
**not** claim that current inference emits the malformed mapping or that this
gap creates false finite support.

Smallest fix: derive the unique rejected parent position, require the replaced
face/failed scan at that position, require all unchanged slot correspondences
to match exactly, and bind the replacement correspondence's CAD/sign to the
child's completed-face/sign fields. Retain the existing exact pose, all-three
support, and whole-origin-ledger checks. Add completion-only, mixed-origin,
and targeted stale-field refusal tests through the live V16 binding seam.

### P2-D3 — Expected alternative-shift assertion is unconditionally true

`tests/test_cad_guided_plane_initialization_v15_completion.py:182` ends with
`or True`. Python therefore makes this entire assertion pass regardless of
the alternative child's status or proposed expected shift. Other assertions
still check two passing distinct groups and the main recovery, so the whole
test is not vacuous; the specific alternative-shift mathematics are untested.

Remove the escape hatch and derive the expected signed shift from the retained
independent fixture and the actual declared axis convention. Do not simply
weaken the test to an absolute shift if the intended property is signed pose
consistency. The near-zero scheduling negative control at `:319` through
`:328` is useful and should remain: it honestly demonstrates that |lambda|
priority can omit the farther correct completion under a small quota.

### P2-D4 — Source test coverage does not yet establish connected completion recovery or fatal-after-success closure

The natural-parent test at
`tests/test_cad_guided_plane_initialization_v15_completion.py:331` through
`:351` establishes (if it passes) that actual search-generated two-of-three
parents are found and every selected child is evaluated. It does not require
a useful completion to survive, bind through V16, and reach measurement. The
direct recovery fixture exercises `_complete_two_of_three` with a constructed
parent. The inspected integration fixture's group at
`tests/test_cad_guided_completion_v1_engine_integration_v1.py:477` still has
ordinary origins; its named tests exercise the inherited context path, not a
new completion-only selected group.

The new preflight test at completion-test `:305` verifies retention before a
row-cap failure. It does not drive the outer initializer after an earlier
completion has passed and a later query/build/serialization fails. Add a
bounded invented-input connected completion-only case and an injected late
failure that verifies retained selected children but no surviving start or
measurement prefix. Existing successful frozen suites need not be rerun merely
because this new test source exists. Root owns freeze and execution decisions.

## Budget and scheduling limitations, separately from defects

The new lane explicitly declares a **separate additional** budget envelope
(`backend/cad_guided_plane_initialization_v15.py:702` through `:711`). Its
query/box calls use lane counters and the same cumulative surface-work
deduction machinery at `:494` through `:550`; selected query row/call demand is
preflighted before heavy support binding. `max_hypothesis_records` is now
enforced at `:848`.

This is not an unchanged total-work experiment: the main-lane envelope plus
completion envelope must both be in the prospective freeze, while the outer
wall/RSS limits remain unchanged. The inherited budget object also exposes
`max_prefilter_unsigned_assignments` and `max_refined_start_groups`, but the
completion enumeration does not independently enforce those lane fields;
parent/child policy caps govern enumeration and the main global refinement cap
governs downstream starts. The new `face_normal_tests`, `candidate_sort_work`,
and `pose_keys` counters are not added to `work_units`. These stages remain
bounded by input and policy caps, but the report should not claim that every
advertised inherited field or all CPU work is charged by the finite-query work
counter. A freeze can state inactive fields and the work formula explicitly;
otherwise use a completion-specific budget schema. This source review does not
establish actual time/RSS capacity.

The refinement schedule now preserves the previous main-lane strata first and
uses only remaining slots for completion-only groups (`:1098` through `:1109`).
This prevents displacement of existing main starts, but can leave zero
completion-only refinement slots when the main quota is full. Its omissions
are recorded. Candidate order is still increasing |lambda| then CAD face ID;
parent order is still hypothesis index. Neither is an exhaustive search or a
proof of pose uniqueness. The label and negative-control test now say so.

The completion method remains a defensible bounded mechanism for the retained
two-of-three pattern. It does not close missing scan planes, null/near-null
axial constraints, wrong rotations, CAD-face split/merge correspondence,
ambiguity beyond the selected quota, or the main lane's previously described
Wahba/heuristic rejection limitations. There is no basis here to replace this
mechanism with the generic cone-feasible witness proposal before testing its
own intended recovery path.

## Envelope edge note and next gate

At `backend/cad_guided_report_envelope_v7.py:179`, a malformed top-level
`evidence` container adds unknown status for ordinary hypotheses/candidates,
but not for completion records. With no direct completion lane, completion
count is reported as zero/no retained list rather than unknown inaccessible
nested structure. This does not affect the valid direct/nested completion-list
fix, but a consistent malformed-container contract should propagate that
unknown status for completion records too. The docstring at `:6` also still
says V6 although only canonical V7 is exempt.

Next gate: Fable's narrow fixes and targeted invented-input evidence, followed
by independent source/test-result closure on exact hashes. Runner, protocol,
measurement, and contact-projection audits are owned by root/other reviewers.
No new physical run is designed or authorized by this document. No new F1,
historical-score improvement, measured bend accuracy, production promotion,
calibrated units, or manufacturing approval is claimed.
