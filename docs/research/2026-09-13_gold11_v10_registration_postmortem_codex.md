# V10 registration/refinement postmortem — independent bounded mechanism review

Written 2026-09-13 UTC (local date 2026-09-14). Reviewer: Codex
`v9_completion_cost_postmortem`. This is complementary to the independent POST,
not a repeat of its custody/scoring audit or a successor PRE.

## Outcome and priority

V10 repaired termination/retention, not registration accuracy. All 22 retained
worker receipts report exit 0, no supervisor budget stop and complete derived
result retention. Registration published a supported context in five slots;
only three also had native features available for measurement. Eleven slots
reported `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS`, but that label must
not be read as exhaustive geometric rejection: ten of those slots also contain
CPU-expired candidates, and some contain unattempted alternatives.

**A layer-separation-only V11 is justified for the narrow objective of a first
nonempty output.** It targets the next failed stage on the three already
measurable slots, without coupling that repair to a new pose/search mechanism.
It is not sufficient evidence toward the broad F1/count targets. The author's
released narrative reports all 43 evaluated windows stopped at the first flange
fit; this review did not independently analyze those flange rows. Fixing that
stage may expose later arc/coverage/identity refusals, so neither a nonempty
prediction nor a positive localized F1 is guaranteed. No registration change
is required by this report before that narrowly scoped successor.

## Scope, quiescence and exact evidence

Read canonical AGENTS, current status, the complete V10 protocol, named frozen
registration/initializer/workflow source and retained case `RESULT.json` /
`COMPLETED.json` ledgers only. A fresh process-table check before derived reads
found no Gold11 benchmark/waiter/scorer, pytest or cad_guided process. Root
separately supplied the observed outer exit at 23:04:19 UTC. No raw scan/CAD,
Gold, SELECTED_REFERENCES, opaque cache or attempt-control content was read.
No geometry, test, scorer, replay or product import executed. JSON projection
and arithmetic used jq; only this new report was written.

Evidence base: exactly 22 RESULT and 22 COMPLETED files under
`docs/research/gold11_mesh_development_v10_attempt/cases/*/*/`.
SHA256 of their **sorted SHA256/path inventory lines**, produced from repository
root by the following read-only command, is
`c260c63bea1c4ab9c66c5a3ebdcedbc5a6bce82f6d4880d57861cce10684d197`:

```sh
shasum -a 256 docs/research/gold11_mesh_development_v10_attempt/cases/*/*/RESULT.json docs/research/gold11_mesh_development_v10_attempt/cases/*/*/COMPLETED.json | LC_ALL=C sort | shasum -a 256
```

Frozen source pins (paths below are relative to the attempt's `source/`):

```text
088c0b5e9fd940094fd44462e7e5c831299fe2c69b87418d09c362772263bae0 backend/cad_guided_global_registration_v3.py
ee05a03eb564c9fcc8c3cee631e4f429d0c66d9ec5468bc932377368105750e7 backend/cad_guided_global_registration_v5.py
7dad12ae569c2044681864bb136105e0fea8750a4c1e454797a9f9fd874cb63b backend/cad_guided_plane_initialization_v16.py
5c4fa6b8f3fb97a23398b9fb00cb418f2f47f0477b93693c6a3c0c4cfd8a5a63 backend/cad_guided_mesh_offline_v17.py
e59aff47a8312bfce44139a0a1977a33dff4a0a86039502b8c34fe5f1f27889b docs/research/2026-09-13_gold11_mesh_development_v10_protocol.md
```

## Exact retained counts — two narrative corrections

The retained ledgers aggregate to **31 completed / 15 expired / 28 omitted =
74 selected candidates**, not 16 expired. The CPU guard is true in **15 of 22
slots**, not 16. All 15 expiries and 28 omissions are CPU-guard dispositions,
not row-quota stops. No candidate failed fatally or remained NOT_ATTEMPTED in
the returned disposition ledgers. These corrections were sent to root for the
author narrative; terminal artifacts remain untouched.

Of the 31 completed candidates, 5 support a conditional context and 26 reject.
The 18 residual-reason occurrences and 9 coverage-reason occurrences **overlap
on one candidate**: they are not 27 distinct rejected candidates. The residual
gate is 1.0 mm; the proximity-coverage floor is 0.75. Returned context p95 upper
values for residual refusals are approximately 1.20–1.62 mm. Several coverage
refusals have acceptable p95 but only approximately 0.46–0.61 coverage. Neither
a low residual on a subset nor high coverage alone establishes a valid pose.

The following table uses short case prefixes only for reporting. They must
never become inference/scheduling features. C/E/O is selected-candidate
completed/expired/omitted; "groups" is all passing main/completion start groups
before lane group quotas. A blank initialization is unknown/unavailable, not 0.

| Case | Control groups; C/E/O | Challenger groups; C/E/O | Supported contexts / consequence |
| --- | --- | --- | --- |
| CG08 | 0/4; 3/1/0 | 0/4; 3/1/0 | 0; residual refusals |
| CG191 | 5/12; 2/1/6 | 5/18; 2/1/6 | 1 each; native features unavailable |
| CG1E0 | 1/2; 2/1/0 | 1/2; 2/1/0 | 1 each; measurement reached |
| CG4BDD | 1/3; 3/1/0 | 1/2; 3/0/0 | control 1; challenger 0 |
| CG4FAF | 17/0; 4/1/7 | 11/0; 3/1/7 | 0; native unavailable also |
| CG64 | 0/2; 0/1/1 | 0/2; 0/1/1 | 0; first candidate expired, native unavailable also |
| CG65 | 0/0; 0/0/0 | 0/0; 0/0/0 | no passing finite starts |
| CG84 | 0/0; 0/0/0 | 0/0; 0/0/0 | 44 completion children each; none pass |
| CGE4B | unavailable; 0/0/0 | unavailable; 0/0/0 | noisy/composite scan-plane proposal unavailable |
| CGEBA | 0/2; 1/1/0 | 0/2; 1/1/0 | 0; coverage refusal |
| CGF41 | 0/2; 1/1/0 | 0/2; 1/1/0 | 0; coverage refusal |

There are 101 passing start groups before quotas and 74 selected candidates.
The 27 unscheduled groups are 22 completion alternatives in the CG191 pair
plus five main alternatives in CG4FAF control. They are distinct from the 28
selected-but-omitted candidates. Thus 70 of the 101 passing groups were either
outside the refinement allocation, expired, or omitted; their final support
is unknown. Passing initialization is not passing whole-part registration.

## What consumed the work

All **31** completed candidates report **30 coarse and 30 fine iterations**,
both at `ITERATION_LIMIT`. Their per-candidate CPU receipts span
193.960664–1422.293850 s; median 416.010668 s. Total candidate CPU:

| Disposition/lane | Completed count / CPU s | Expired count / CPU s | Omitted count |
| --- | --- | --- | --- |
| Main | 15 / 6115.490128 | 4 / 730.771528 | 18 |
| Completion | 16 / 10849.063718 | 11 / 4929.769151 | 10 |
| Total | 31 / 16964.553846 | 15 / 5660.540679 | 28 |

These are disjoint per-candidate receipts; omitted rows have no candidate CPU
receipt, not a claim of zero surrounding overhead. Completion-prefilter CPU
adds 19.803771 s, giving metered lane totals of 6846.261656 s main and
15798.636640 s completion. Do not add nested initialization/evidence copies
of the same receipts again. Process preparation, publication and serialization
are outside these candidate totals. No profiler attribution within an expired
candidate is available from its final row: coarse/fine completion counters were
not published, so the precise subphase must remain unknown.

The completion prefilter is not the present dominant cost. Sequential finite
surface refinement is. Nor do these ledgers establish completion starving
otherwise runnable main starts: main is scheduled first (V16:1351–1382), and
the completion-heavy cases mostly have zero main passing groups. Conversely,
CG191 runs main work until the guard and never refines its selected completion
alternatives. Lane order and long serial candidates do restrict breadth;
whether an unvisited alternative would succeed is unobserved.

For the eleven no-context slots, the retained subset contains 20 completed
rejected candidates, 10 expired and 16 omitted. CG64's two slots have **no
completed candidate at all**, so their generic no-supported-context reason is
an honest empty-publication outcome, not proof that any pose failed geometry.
CG4BDD challenger is different: all three selected candidates completed and
rejected without tripping the guard. More CPU cannot be assumed to cure that
already-completed selected set.

## Mechanism interpretation and two future priorities

1. **Accelerate and fairly stage proposal refinement; keep final validation
   unchanged.** Frozen V3:336–402 uses point-to-point materialized closest-point
   updates, one finite surface query per iteration and a motion termination
   threshold of `1e-8` in millimetre coordinates (398–401). Every completed
   physical candidate hitting 60 iterations is direct evidence that this
   termination path did not save work. It is not proof that the last iterations
   were unnecessary or that increasing the threshold is safe. A justified next
   research seam is deterministic staged/multiresolution candidate refinement:
   give each allocated start a small initial work tranche, resume promising and
   diverse starts under frozen work budgets, and use a declared model-aware
   optimizer convergence rule. Cheap/approximate proposal work must never
   replace the unchanged bound-aware final full-row support query or make
   omitted alternatives rejected. Preserve fatal-versus-typed-expiry handling.

   Independent synthetic acceptance: constructed partial and symmetric poses
   where the useful start is late; a cheap wrong local minimum followed by a
   recoverable start; thin-sheet and trimmed-edge scenes; near-gate final p95
   and coverage. Require deterministic work allocation, exact disjoint unknown
   sets, no partial accepted pose after fatal state, final geometric validation
   independent of the proposal approximation, and measured work reduction at
   a matched correctness/error envelope. Include a case where early stopping
   would be wrong. Do not optimize a schedule using physical case names or Gold.

2. **Stop unavoidable native-feature failures before expensive registration in
   the ordinary workflow; keep a separately declared diagnostic path if needed.**
   V17 extracts/stores native status at 586–587 but does not apply the
   unavailable/empty-feature refusal until 682–684, after sampling/registration.
   Six slots already had `INVALID_NATIVE_SOLID`; their metered registration
   lanes nevertheless consumed **8914.695307 s**, about 39.4% of all metered
   lane CPU. That work cannot produce measurement in this workflow even when
   a context succeeds, as the CG191 pair demonstrates. This is a source-proven
   ordering inefficiency, not a reason to suppress failed cases or discard
   diagnostics. A successor may fail early with an explicit stage disposition,
   retaining native evidence and declaring later stages not attempted. Test
   unavailable/empty features with a registration sentinel that must never run;
   valid/PARTIAL features must still traverse the real binder/measurement path.
   This saves runtime, not F1, and need not be coupled to V11's layer mechanism.

V5:504–523 establishes the current proximity-based admission contract: all
selected scan rows form the coverage denominator, p95 is computed over definite
proximity rows, and the result explicitly does not certify pose/correspondence.
The retained failures do not distinguish wrong pose, partial visibility,
deformation, sheet-side mismatch and acquisition contamination sufficiently
to justify loosening its gates. A future visibility/material-aware model would
need independently constructed evidence, not a threshold increase to admit
the observed 1.2–1.6 mm cases. No nominal CAD answer or observed bend count may
be inferred from these proposals.

## Limits and next handoff

No physical measurement truth or calibration was added. The common all-null
score is supplied by the released narrative/root scoring audit, not recomputed
here. Historical 0.4580152672 remains non-comparable. Candidate/lane costs are
now observed in V10, but they do not retrospectively identify which V9 lane
caused its kills or establish isolated speedup from V10's combined changes.
The unavailable CGE4B scan-plane metadata is encoded; this bounded review
retains its explicit initializer reasons and does not infer its deeper cause.

Proceed with the bounded layer-separation successor if its independent
implementation/tests/PRE qualify it. Retain the registration/cost deficiencies
above as subsequent mechanism work, with exact narrative counts corrected.
Do not reopen or rescore V10. Fable remains the sole engine/test/runner/status
editor; no implementation was made or requested by this report.
