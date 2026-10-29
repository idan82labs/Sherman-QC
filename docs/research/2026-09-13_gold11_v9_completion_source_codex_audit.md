# V9 completion source audit — initial V15 snapshot not greenlit

Date: 2026-09-13. Independent Codex source-only review. Fable 5.1 owns all
engine/test edits. This report changes no source or prior audit.

## Verdict

**Not greenlit for freeze/PRE at the inspected snapshot.** The completion
mechanism has a correct nominal axial-shift calculation and rechecks all three
finite supports, but one connected workflow blocker prevents newly recovered
poses from reaching measurement. Failure/report retention also needs closure.
These are concrete source findings, not predictions of a physical F1 result.

No imports, tests, numerical probes, physical inputs, Gold, attempts, controls or
private caches were accessed. Reviewed the full new completion test source and
the complete V14-to-V15 source diff, then narrowly traced the workflow origin
validator, report envelope and budget declarations. The newest checkpoint was
03:38 UTC. Test passes recorded by the author were not rerun or independently
verified here.

## Exact reviewed files

```text
ce4a61af90db1339ace0914cd359ea6e0afdf22dd5b9f038c1e2d9d7f1eeaba4  backend/cad_guided_plane_initialization_v15.py
9fd6c11e987cd260e0247154b901ebc67e600e0dba5766787a22b907ffbf2378  tests/test_cad_guided_plane_initialization_v15_completion.py
441c618d4998edaca0d401dd22f14d1444f248750a807386fecef19741708769  backend/cad_guided_mesh_offline_v16.py
c7cc45a362c5972aff78265b2ce63fa4d41ec023b662640d7d197c23003ba28e  backend/cad_guided_report_envelope_v6.py
```

These hashes were rechecked during the audit and still matched. Findings below
bind these snapshots; subsequent edits require a delta review, not an implicit
acceptance by this document.

## P1-1: new completion-only poses fail the maintained workflow origin gate

V15 creates a new exact start group with `hypothesis_indices=[]` and a separate
`completion_indices` list (`cad_guided_plane_initialization_v15.py:801–812`).
That is a reasonable separate-origin representation, but the connected consumer
has not been updated for it:

- `cad_guided_mesh_offline_v16.py:209–211` requires a **nonempty** ordinary
  `candidate['hypothesis_indices']`, otherwise raises
  `MISSING_SELECTED_START_SUPPORT_ORIGINS`.
- Its following checks accept only main TRIPLET/TWO_PLANE_CONTEXT origins
  (`:212–229`); they do not inspect or validate completion origins.
- `_registration_binding` is called before generated-context measurement
  (`:489` onward), after a context survives registration.

Therefore every genuinely new completion-only pose that reaches this boundary
is refused before measurement. A completion coinciding with an already passing
main pose may get through on that main origin; this cannot demonstrate recovery
of a previously unavailable pose. This blocker directly defeats V9-A's intended
benefit even if its local tests and refinement pass.

Required fix: a closed, explicit completion-origin branch at the binding seam,
validating the complete selected group's ordinary and completion origins,
parent provenance, exactly three refreshed passing supports, replacement
face/sign, exact initialized pose and lane identity. Do not populate the empty
list with an unrelated main hypothesis merely to satisfy the old validator.
Retain rejection of spoofed/stale/missing completion origins.

Required regression: a completion-only group with no ordinary passing origin
must traverse actual registration binding into the measurement path; mutations
of completion index, parent, third face, support hash or initialized pose must
refuse. The present test file never exercises this seam.

## P1-2: selected completion records disappear on preflight/build/query failure

V15 builds `candidate_rows` privately, but publishes each row into
`lane['hypotheses']` only **after** `_evaluate_completion_supports` returns
(`:774–796`). The selected denominator and each parent's completion indices
are published earlier (`:764–770`).

Consequences visible directly from control flow:

- Failure during support binding/preflight/index build (`:777–792`) leaves
  `scheduled_completion_count > 0` but **zero** retained child records.
- Failure during a later finite query loses the current child's attempted
  evidence and every selected-but-unattempted child, retaining only completed
  predecessors. Parent indices then point at absent completion records.

The main V14/V15 lane deliberately publishes its full ledger before work for
exactly this reason. The completion lane must do the same: publish selected
records before validation/query work, mutate their status in place, and preserve
current incomplete and later unattempted dispositions on fatal exit. If a
generation failure occurs after a partial selection, retain the known prefix
as attempted-generation evidence, not as a complete selected schedule.

**No favorable-prefix promotion was found:** `_initialize` calls the old
failure helper, clears `start_groups`, sets completion flags false, nulls poses
in all passing groups and returns no context (`:1059–1074`). This protects the
engine output. However the completion lane's already passing rows are not
explicitly marked invalidated by the old helper, which visits only main
`result['hypotheses']`. Add an explicit lane failure scope/usable=false and
attempted-count distinction rather than leaving ambiguous `PASSED` prefixes
under an incomplete lane.

Regression cases should fail before the first query, during the first support,
and after one passing child. Every selected identity must remain with honest
disposition, and no start/context may survive the fatal failure.

## P1-3: report-cap refusal drops known completion counts

V15 retains the frozen V6 report-envelope helper (`:57`, used at `:1073`).
`cad_guided_report_envelope_v6.py:_retained_evidence` counts ordinary
`hypotheses`, either direct or under `evidence.initialization`, plus selected
registration candidates (`:160–185`). It never examines
`completion_lane.hypotheses`.

On a detailed-report cap/serialization refusal, all completion detail is
discarded but those known records are absent from the discarded-hypothesis
accounting. The helper's old schema only knows the old lane; this is an
integration gap, not permission to edit that frozen helper.

Use a new versioned additive envelope/binding appropriate to separate
completion identity. Count known records without double-counting aliases;
preserve the first refusal and idempotence, and use unknown/lower-bound status
where container evidence is malformed. Test direct initialization, wrapped
registration, repeated-envelope and zero-cap cases with completion records.

## P2-1: replacement normal sign is fixed to the old face

At `:712`, V15 defines `n3w=old_sign*(R@n3)`, accepts a new face only if its
oriented normal points toward that value (`:725`), and copies `old_sign` into
the replacement correspondence (`:753–757`). It never chooses the new face's
sign independently. An otherwise compatible unoriented replacement on the
opposite stored-normal branch is excluded before finite support is tested.

Important source limitation: the CAD fitter canonicalizes normals by their
largest-magnitude component (`cad_guided_mesh_plane_proposals_v4.py:223`), so
exactly parallel production faces normally have the same orientation. Merely
flipping an arbitrary internal dictionary does not prove a reachable physical
failure. A real discontinuity remains near that canonical-sign boundary:
nearby normals around `(1,-1,0)/sqrt(2)` can select different dominant
components and acquire opposite canonical signs despite a tiny unoriented
angular gap. Such a case should be constructed through the real CAD fitter.

Compute `q=R@n3`, select/validate a candidate-specific sign `s_j`, and compare
`q` to `s_j*m_j` using the existing allowance; update both the correspondence
sign and the child's sign list. The shift formula may continue to use the
unsigned target equation because its sign cancels from numerator/denominator.
Do not turn the compatibility gate into a new wider threshold.

Add a sign-boundary fixture and a consistent plane-equation sign-flip unit
test, both retaining the same finite geometry and expected pose. No physical
recovery is inferred from this source finding.

## P2-2: the completion budget object advertises unused hard fields

The protocol explicitly declares an **additional separate** completion budget
(`2026-09-13_gold11_mesh_development_v9_protocol.md:122–125`), and the new
runner binds it separately. Consequently a fresh completion query ledger is
not by itself an undeclared reset; this differs from the earlier tentative
design's “same cumulative caps” wording. The increased total allowance must
remain clear: up to the base plus completion caps, with the unchanged outer
CPU/RSS limits. No started budget is changed here.

But `_completion_configuration` accepts the whole V12 budget object, including
zero-valued smaller caps (`:213–223`), while the completion path never enforces
its `max_hypothesis_records`, `max_prefilter_unsigned_assignments`, or separate
`max_report_bytes`. Child count is instead controlled only by
`CompletionLanePolicyV1.max_completions_total`, whose allowed maximum is 65,536
(`:96–110`). Report bounding uses only the **main** report cap (`:1073`).

Thus callers can supply a smaller published completion hard cap that is
silently ignored. The current default policy's 4,096 children is below the
default 8,192 record cap, so this particular mismatch does not prove an
observed default-run overrun. It is a public configuration-contract bug and a
missing small-cap test.

Either enforce the relevant fields with preflight or introduce a narrow,
accurately named completion budget type and declare unused fields inactive.
Do not claim an unsigned-assignment cap using the child's copied parent
`unsigned_assignment_id` (`:756`): changing the third face changes the actual
assignment. Rename that as a parent-origin field or establish a distinct
completion-assignment namespace, while preserving `completion_index`.

The query statistics do accumulate within the completion lane and the helper
enforces query/build resource caps. However parent-face enumeration and sort
work (`:687–744`) have no charge/counter beyond eventual candidate categories;
they are bounded indirectly by parent/face limits, not the published numerical
work count. Repeated `_bound_scan` checks for every child occur **before** the
planned row/call preflight (`:777–790`), so an over-cap candidate set can repeat
large support validation/hash work before being refused. Deduplicate validated
scan/face bindings within the phase and preflight selected raw populations
before repeated heavy validation. Record the bounded enumeration/sort cost
separately; do not call the existing query-work counter total completion work.

## P2-3: tests do not establish completion-only connected recovery or failure safety

The test fixture runs real extraction/search/prefilter, which is useful. But
it requires an already passing ordinary triplet (`test…completion.py:102–106`),
and the general ledger test explicitly requires a main passing group (`:124`).
It does not require the natural completion parent population to be nonzero.

The recovery tests then fabricate a two-of-three parent from that passing
triplet, changing a face, pose and status fields (`:131–149`), and invoke the
private completion helper directly (`:153–178`). This validates selected
geometry behavior with real finite queries, but not naturally generated
two-of-three evidence, downstream registration binding, or the full workflow.
The synthetic parent’s old PASS flags/digests are not fully recomputed before
the helper call; that is test setup, not a faithful replay of a completed
parent contract.

Missing checks corresponding to this audit: completion-only origin binding;
fatal prefix/selected-denominator retention; report-envelope completion counts;
candidate-specific signs; advertised small hard caps; and shift-induced loss
of one formerly passing finite support. These can be focused inert synthetic
checks, not a broad physical readiness campaign. Keep the existing direct
helper tests, but label their evidence scope accurately.

## P2/P3 schedule and diagnostic limitations

The declared |shift| then face-ID ranking is deterministic and nonexhaustive;
it does not claim globally complete pose search. It can still spend the quota
on near-zero wrong/copanar faces while omitting a farther correct completion.
The existing omission test verifies counts, not geometry under adverse order.
Add the farther-valid/coplanar-ID-permutation control from the design review,
or retain this explicitly as a known finite-schedule limitation.

Two small diagnostic issues are visible:

- A parent can be labeled `SCHEDULED` because its per-parent `scheduled_here`
  list is nonempty even when the total cap omits **every** one of its children
  (`:746–768`). Use the actual `scheduled_count` to distinguish it.
- `NO_COMPATIBLE_THIRD_FACE` is also used when normals were compatible but every
  denominator was unresolved or every shift exceeded the cap (`:725–768`).
  Counters permit reconstruction, but the status wording should not imply
  mathematical incompatibility. The current axis-parallel test asserts that
  misleading label rather than a distinct unresolved disposition.

## Mathematics and safety checks that are already sound in scope

For the current zero-offset CAD planes, the code computes

`shift = -m_j dot (R*w_s+t-a_j) / (m_j dot u)`

(`:711–733`). This is the correct nominal axial shift putting source witness
`w_s` on the candidate infinite plane. Units are mm in the numerator, a
dimensionless denominator, and mm for the shift. Using unsigned target normals
is valid because negating the whole target equation cancels in the quotient.
The sign-filter finding above is a proposal-coverage defect, not a sign error
in this quotient for candidates that are actually admitted.

The pair cross product is normalized and below-floor pairs are unresolved
(`:699–706`); below-floor third-axis projections and shifts outside the explicit
limit are counted (`:728–735`). A proposal does not publish calibrated
uncertainty or claim exact whole-plane alignment under mismatched normals.
The default floors and maximum shift are visible policy, not hidden tuning.

The scan normal is renormalized again while its offset is left unchanged
(`:709–711`). The real scan producer already returns a normalized eigenvector
(`cad_guided_noisy_mesh_planes_v5.py:266`), so this is ordinarily an ulp-scale
inconsistency, not an established material failure. Prefer the canonical
normal-plus-offset equation or normalize both coefficients consistently, and
test nonzero offsets. No large offset/unit bug was found in the connected
normal producer.

Most importantly, `_fresh_support` starts new evidence, and
`_evaluate_completion_supports` transforms, box-screens and (where required)
queries **every row of all three supports** (`:575–644`). It does not reuse the
parent’s two passing flags as child evidence. It retains the same lower-bound
coverage acceptance and does not promote a box proxy. The first two CAD face
IDs are excluded from replacements (`:720–722`). Passing children join the
exact groups before the original bounded refinement schedule (`:1002–1005`).

## Handoff

Fix the connected completion-origin gate and both evidence-retention seams
first; otherwise a run can spend additional CPU yet still reject every newly
recovered context or lose the precise failed-stage denominator. Then close the
sign/configuration/test findings with focused source-level controls, rehash the
changed closure and obtain independent delta acceptance before freezing.

This report neither requires generic witness fallback nor authorizes another
physical run. V8 and earlier attempts remain terminal. No historical-score,
calibration, production or physical-recovery claim is made.
