# Gold11 full11 V9 — result narrative (clean run, observed exit 0, no measurement output, F1 undefined)

Date: 2026-09-13. Author: Fable 5.1 worker `1519fdc2` (sole implementation
owner). Evidence: the terminal V9 attempt's DERIVED records only
(`SCORED.json`, `INFERENCE_TERMINAL.json`, `COMPLETED.json`, `FINAL_GATE.json`,
per-slot `COMPLETED.json`/`OUTPUT.txt`/`RESULT.json`), the outer launch
receipts beside the attempt directory, and the frozen source. No raw scan/CAD,
Gold, cache or credential was opened. This narrative claims no score: the
unchanged scorer returned null and null stays null. V9 is terminal; it will
not be re-scored, resumed or reopened. Independent POST acceptance is recorded
in the last section when available.

## Headline

The frozen V9 successor (bundle `c50a8cb9…4b47d`, PRE GREEN) ran exactly once
through the bound waiter. The waiting parent OBSERVED the outer return code 0
(started 05:36:07 UTC, finished 13:34:04 UTC, parent wall 28675.6 s ≈ 7 h 58
min, 22 durable case/arm records, parent lifetime peak RSS 1.26 GB, sampled
leaf-tree peaks 0.49–1.95 GB, ps-monitored tree peak 1.97 GiB; every limit
untouched; `failures []`). Both arms score **TP 0 / FP 0 / FN 71, recall 0,
precision null, localized F1 null**; emitted support lines 0; all 22 slots are
in the denominator and all 71 references are retained false negatives. No
numeric F1 exists, no improvement over V8 or V7 is claimed, and the historical
0.4580152672 remains non-comparable. Every V9 outcome is a COMBINED
consumed-development change (completion lane, kernel-V2 selection and
diagnostics, derived δ/e and 30° support policies) on the unchanged V8
mechanisms; the in-run arm comparison is scan-plane lane/schedule only on the
same V16 engine.

Two mechanisms account for the empty output:

1. **Fourteen slots were terminated by the per-case 1860 s CPU cap inside
   surface V4 refinement** (exit code −24, SIGXCPU; wall 1861–1862 s each):
   CG08506, CG191, CG1E0, CG4FAF, CG64, CGEBA and CGF41 in both arms. V8 had
   four such slots (CG191, CG4FAF). The sampled stacks at 1770–1861 s sit in
   `cad_guided_triangle_surface_v4._paired_candidates` / `leaf` / `walk` and
   the V3 distance bounds, i.e. the competitive-candidate surface queries of
   whole-part refinement. These slots returned no workflow stages
   (`NO_RETURNED_WORKFLOW_INTERNAL_STAGES_UNKNOWN`), so nothing downstream of
   refinement is known for them.
2. **The one slot that reached MEASUREMENT (CG4BDD control) was refused by a
   V9 wiring defect**: `SUPPORT_COVER_MEASUREMENT_FAILURE:ValueError:
   INVALID_SUPPORT_COVER_FEATURES`. Source-confirmed cause: mesh offline V16
   imports `CADGuidedFeature` from the V1 kernel `cad_guided_measurement`
   (line 33) and builds the native features with that class (line 692), while
   support cover V3 (line 144) requires `type(feature) is` the kernel-V2
   `CADGuidedFeature` re-exported by context V2. V8's chain was consistent
   (V15 built V1 features for cover V2, which requires the V1 class). In V9
   every real measurement dispatch would refuse this way; only CG4BDD control
   reached that phase. The connected-chain test constructed kernel-V2 features
   directly and the engine integration test mocks native features as a
   sentinel object and mocks the cover call, so no test exercised V16's actual
   feature construction into cover V3. This is the author's defect; it is
   recorded here as such and is the first item of the V10 diagnosis.

## Outcome taxonomy, V8 → V9 (22 slots each)

| Immediate cause | V8 slots | V9 slots | V9 cases |
| --- | ---: | ---: | --- |
| CPU termination (1860 s) inside surface V4 refinement | 4 | 14 | CG08506, CG191, CG1E0, CG4FAF, CG64, CGEBA, CGF41 (both arms) |
| Completed prefilter, no passing finite start | 12 | 4 | CG65 (no two-of-three parent, as predicted), CG84 (1 parent, 44 children scheduled, 0 passing) |
| Scan-plane proposals unavailable (frozen membership cap) | 2 | 2 | CGE4B (both arms) |
| All start groups rejected at the 1.0 mm context gate | 1 | 1 | CG4BDD challenger (main 1.196 mm; completions 1.318 / 1.338 mm) |
| Admitted context, all measurement windows refused | 3 | 0 | — |
| Admitted context, measurement dispatch refused by the feature-class defect | 0 | 1 | CG4BDD control (main group 0.666 mm admitted; completions 0.738 / 0.720 insufficient proximity, 1.306 outside gate) |

The eight returned slots refused exactly where V8 refused for the same cases,
except CG4BDD control, which V8 had admitted and then refused at the flange
gates (43 windows) and V9 now refuses one step earlier at the cover grammar.
The two V8 admitted contexts that V9 does not reproduce (CG1E0 both arms) are
among the fourteen CPU terminations: the completion lane's additional passing
start groups enlarged the refinement schedule, and the per-case CPU cap bound
before those contexts could be evaluated. Whether CG1E0 would have been
admitted again is therefore unknown, not negative.

## Per-slot ledger (wall = leaf receipt wall; RSS = sampled leaf-tree peak, not continuous)

| Case | Family | Arm | Wall s | Peak RSS MiB | Exit | V9 cause | Start groups (main + completion) / lane |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| CG08506AE97D | 10839000 | C | 1861.4 | 1718 | −24 | CPU kill in refine | unknown (no stages returned) |
| CG08506AE97D | 10839000 | H | 1861.0 | 1950 | −24 | CPU kill in refine | unknown |
| CG191DEC7F81 | 49001000 | C | 1861.3 | 1344 | −24 | CPU kill in refine | unknown |
| CG191DEC7F81 | 49001000 | H | 1861.3 | 1528 | −24 | CPU kill in refine | unknown |
| CG1E0BBA2933 | 38172000 | C | 1861.0 | 1556 | −24 | CPU kill in refine | unknown |
| CG1E0BBA2933 | 38172000 | H | 1861.6 | 1735 | −24 | CPU kill in refine | unknown |
| CG4BDD1371B6 | 37361005 | C | 1626.8 | 1400 | 0 | MEASUREMENT_REFUSAL: INVALID_SUPPORT_COVER_FEATURES | 1 + 3; lane 3 parents, 64 scheduled, 3 passing; context [0] admitted at 0.666 mm |
| CG4BDD1371B6 | 37361005 | H | 702.2 | 1540 | 0 | NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS | 1 + 2; lane 1 parent, 18 scheduled, 2 passing; all three rejected at the gate |
| CG4FAF273093 | 49024000 | C | 1861.8 | 490 | −24 | CPU kill in refine | unknown |
| CG4FAF273093 | 49024000 | H | 1861.8 | 516 | −24 | CPU kill in refine | unknown |
| CG64C412CC73 | 40500000 | C | 1862.0 | 1011 | −24 | CPU kill in refine | unknown |
| CG64C412CC73 | 40500000 | H | 1862.0 | 1173 | −24 | CPU kill in refine | unknown |
| CG65CD575B1C | 38172000 | C | 30.8 | 1519 | 0 | NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS | 0 + 0; lane 0 parents (8160 hypotheses, 2045 evaluated) |
| CG65CD575B1C | 38172000 | H | 47.1 | 1823 | 0 | NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS | 0 + 0; lane 0 parents |
| CG84C8FA12D2 | 10839000 | C | 42.8 | 1689 | 0 | NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS | 0 + 0; lane 1 parent, 44 scheduled, 0 passing, 68 normal-incompatible |
| CG84C8FA12D2 | 10839000 | H | 63.7 | 1855 | 0 | NO_FINITE_PREFILTERED_NOISY_PLANE_STARTS | 0 + 0; lane 1 parent, 44 scheduled, 0 passing |
| CGE4B0549E41 | 10839000 | C | 12.5 | 1619 | 0 | NOISY_SCAN_PLANE_PROPOSALS_UNAVAILABLE | lane not attempted |
| CGE4B0549E41 | 10839000 | H | 12.7 | 1673 | 0 | COMPOSITE_SCAN_PLANE_PROPOSALS_UNAVAILABLE | lane not attempted |
| CGEBA9A869AC | 38172000 | C | 1861.9 | 1427 | −24 | CPU kill in refine | unknown |
| CGEBA9A869AC | 38172000 | H | 1861.9 | 1561 | −24 | CPU kill in refine | unknown |
| CGF41F49066A | 38172000 | C | 1861.9 | 1571 | −24 | CPU kill in refine | unknown |
| CGF41F49066A | 38172000 | H | 1862.0 | 1688 | −24 | CPU kill in refine | unknown |

## What the completion lane did (returned slots only)

- CG4BDD control: the main lane passed 1 of 4938 evaluated hypotheses (one
  main group, as in V8); the completion lane found 3 two-of-three parents,
  scheduled 64 children (140 normal-incompatible candidates, 0 omitted), and 3
  children passed the full rescreen/requery, forming 3 new start groups
  (refinement schedule: 1 main + 3 completion, none omitted). All 4 groups
  were refined; the main group was admitted at inlier-plane p95 0.666 mm
  (V8: 0.6659) and the three completion groups were rejected at the context
  stage (0.738 and 0.720 mm `INSUFFICIENT_CONDITIONAL_CONTEXT_PROXIMITY`,
  1.306 mm `CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE`).
- CG4BDD challenger: 1 main + 2 completion groups (1 parent, 18 scheduled, 2
  passing); all three rejected at the 1.0 mm gate (1.196 / 1.318 / 1.338 mm).
- CG84: 1 parent, 44 children scheduled, none passed (68 normal-incompatible
  candidates); CG65: no parent (as the diagnosis predicted); CGE4B: the lane
  was never reached.
- The 14 CPU-terminated slots returned no lane ledger. Their refinement load
  was larger than V8's because every passing completion child is a new start
  group in the unchanged 48-group schedule; the diagnosis's exact-box
  feasibility (completions in CG08506, CGEBA, CGF41, CG64) could not be
  observed at the row level.

So the completion mechanism produced passing finite starts where V8 had none
on this case (CG4BDD: three new groups in one arm, two in the other) and none
of those starts survived the unchanged provisional 1.0 mm admission gate; on
the other cases it either found no parent, passed no child, or was cut off by
the CPU cap. This is consistent with the lane's design as an additive proposal
mechanism, not evidence for or against its value on the terminated cases.

## What V9 did not get to test

The kernel-V2 layer selection, the derived δ 0.03 / e 0.04 measurement policy
and the 30° finite-support policy were never exercised on cohort data: no slot
reached a measurement window, because the only admitted context refused at the
cover grammar before any window was cropped. Their diagnostics, the
two-skin hypothesis, and the per-window residual distribution therefore remain
untested on the physical cohort. They are neither confirmed nor refuted by V9.

## Resources and custody

Parent wall 28675.6 s (campaign limit 104400 s), parent lifetime peak RSS
1262747648 B, sampled leaf-tree peaks 490–1950 MiB against the 8 GiB worker
limit, coordinator HWM and 12 GiB outer limits untouched; the 14 CPU
terminations are retained negative outcomes under the frozen 1860 s per-case
CPU cap, never a cap extension. Scoring ran only after `INFERENCE_TERMINAL`
(22 records); scorer result SHA `f561f65c…b63d` equals `FINAL_GATE.json` and
`SCORER_OUTPUT.txt`; `FINAL_GATE.json` status `PENDING_INDEPENDENT_POST_AUDIT`;
final campaign inventory SHA `4c63a73d…b594` (284 entries). Outer exit record
`docs/research/gold11_mesh_development_v9_attempt_outer_exit.json`:
returncode 0, `OBSERVED_BY_WAITING_PARENT` (waiter 52251, runner 52252), the
evidence V8 lacked.

## Consequences for the next successor (diagnosis to follow, not started here)

1. Feature-class binding: V16 must build (or convert to) the kernel-V2
   `CADGuidedFeature` that context V2 / cover V3 require, with a test that
   drives V16's real native-feature construction into the real cover V3 entry
   (no mocked features, no mocked cover call). One line of wiring plus the
   missing test; the derived measurement policy and layer selection stay as
   frozen in V9 unless their own review says otherwise.
2. CPU-cap terminations: the enlarged refinement schedule (main plus
   completion groups) exhausts 1860 s CPU inside surface V4 competitive
   pairing on seven cases. Candidate mechanisms for review: a declared
   refinement work budget per start group with published omissions (no cap
   extension), or the finite-start audit's deferred items; the choice needs
   its own diagnosis on the retained V9 records and an independent review.
3. Context gate: every completion start that was refined landed at 0.72–1.34
   mm, above or near the provisional 1.0 mm admission; the gate itself is not
   changed by this narrative.

## Independent POST acceptance — 2026-09-13 14:03 UTC

Independent non-author POST (`independent-ai-post-auditor-v9`, read-only;
written to job tmp, copied verbatim with hash verification):
`docs/research/2026-09-13_gold11_mesh_development_v9_post_run_audit.md`,
64734 B, SHA256 `bcdd7e4b414d308ca347f40ddda5c78fb53d24f623ae6b6083bb25c1f0b33dca`.
Decision: ACCEPT the execution custody of the terminal V9 campaign and its
complete negative scoring ledger (engineering audit only), blocking findings
none. The outer exit is OBSERVED (returncode 0 by waiting parent 52251 for
runner 52252), closing V8's evidence limitation; every FINAL_GATE digest,
105 custody files, the 282-entry inventory, both publication inventory
digests and the scoring-request digest were reproduced; 85/85 frozen sources
hash-identical in working tree, snapshot and archive; no retry, resume or
relaxed limit. Findings retained here: 14 CPU kills inside the V15
registration refine/diagnose loop (stages unknown beyond 30 s stack samples;
campaign inference wall 28601.6 s versus V8's 9298.5 s, 3.08x, not a
speed-up); the new outcome class MG (CG4BDD control refused by cover V3's
exact-type guard because V16 passes kernel-V1 feature objects from native
adapter V1), so kernel V2, context V2, cover V3, layer selection, the δ/e
policy and the 30° gap were never exercised on physical data and the identity
counts are null, not zero; completion lane exercised on the six returned
initializations (3+1+0+0+1+1 parents, 64+18+0+0+44+44 children, 3+2+0+0+0+0
passing, all five new CG4BDD groups rejected); main-lane prefilter and
context V4 reproduce V8 exactly on the three comparable initializations; the
measurement-context schedule was not exercised; 84 of the 85 frozen sources
are untracked in git (custody rests on the hash-verified snapshot and
archive); `PRE_RUN_AUDIT.json` is not canonical JSON (both digests
reconcile); outer records are mode 0644. Codex's separate scoring-ledger seam
(`87e7f905…5b9ac`, ACCEPTED_NEGATIVE_SCORING_LEDGER_WITH_LIMITED_SCOPE) and
completion-cost postmortem (`5edc6ee8…5618f`) are integrated in the V10
diagnosis. V9 is terminal.

## Correction (append-only) — 2026-09-13 13:42 UTC: CPU-kill attribution is unproven

Codex's independent ledger review confirms the CG4BDD control refusal as a
contract failure (not scan evidence). Its cost reviewer source-proves the V15
order of work: the completion lane's own prefilter (box screen and finite
queries for every scheduled child) runs BEFORE main-lane refinement, and ALL
refinements (main and completion groups) run before any publication. The
"main strata first" quota therefore orders the refinement list only; it does
not protect the per-case CPU budget, and a slot killed during refinement loses
the already-passing contexts it may have reached. The sentences above that
attribute the fourteen CPU terminations to the completion lane's enlarged
refinement schedule are hypotheses, not established causes: those slots
returned no lane ledger, no stage evidence and no refinement receipts, so the
share of CPU spent in the completion prefilter, in the additional group
refinements, or in the same competitive-pairing site that killed CG191 and
CG4FAF in V8 is unknown. Lane attribution of the kills remains unproven until
a successor publishes per-stage CPU receipts before refinement. The separate
cost report `2026-09-13_gold11_v9_completion_cost_codex_postmortem.md` and the
scoring-ledger report `2026-09-13_gold11_v9_scoring_ledger_codex_post_audit.md`
are inputs to the V10 diagnosis, which starts only after the independent POST.
No V9 record is edited or re-scored.


## Correction (append-only) — 2026-09-13 13:59 UTC: defect location and CG84 wording

Codex's reading is right and the attribution above is wrong on the module:
the kernel-V1 `CADGuidedFeature` import (line 33) and the feature construction
(line 692) are in `backend/cad_guided_native_features_v1.py`, the native
feature extractor that mesh offline V16 calls; V16 passes those V1-class
objects to support cover V3 (V16 line 621), which requires the kernel-V2 class
(cover V3 line 144). The defect is the producer/consumer type mismatch between
native features V1 and cover V3 across the V16 wiring, not a line inside V16.
Second, CG84 must not be read as passing-start progress: its completion lane
enumerated one parent and 44 children and ZERO children passed the rescreen
(68 normal-incompatible candidates); the only case with new passing completion
starts was CG4BDD (three in control, two in challenger), all of which were
rejected at the context gate. Both reports read for the V10 diagnosis:
`2026-09-13_gold11_v9_scoring_ledger_codex_post_audit.md`
(87e7f905e3a33c18b13e6b2261f7c545b9ff4e18102e2cf39a21c5a707a5b9ac) and
`2026-09-13_gold11_v9_completion_cost_codex_postmortem.md`
(5edc6ee890310eb4b22b2d367077affe423a608cae96266fe675adb07fb5618f).


## POST addendum (append-only) — 2026-09-13 14:10 UTC

Reviewer addendum (same non-author auditor, read-only, copied verbatim with
hash verification): `docs/research/2026-09-13_gold11_mesh_development_v9_post_run_audit_addendum.md`,
9420 B, SHA256 `67237db70c3d2232c62c1ce0ca72e88c21011ce424a5d18ef25a48cf27362a2e`.
Three corrections to the original POST (which stays verbatim): (1) archive
MEMBER hash identity is withdrawn — only the tar container hash, the listing
(92 entries = 6 directories + 86 files = FROZEN.json + 85 sources) and member
SIZES were verified; (2) 10 of 11 case pairs (20 of 22 slots) share the same
cause across arms, the exception being CG4BDD (control MG versus challenger
RG); (3) the worker CPU comparison is stated both as unpaired totals (V8
1850.329 s over 18 returned workers versus V9 2535.780 s over 8) and paired
over the 8 slots returned in both runs (V8 763.757 s → V9 2535.780 s, the
increase entirely in CG4BDD, 541.884 → 2326.930 s). ACCEPT stands with the
same limits. Correspondingly, the per-slot table above compares V9 to V8 by
cause per slot, not by paired CPU.
