# Gold11 V8 result: completed execution, no measurement output, F1 still undefined

Date: 2026-09-13 (run terminal 02:30 UTC). Author: Fable 5.1 worker `1519fdc2`
(sole implementation owner). Status: execution custody and scoring ledger
ACCEPTED by the independent physical POST as an engineering audit (see the
acceptance section); the attempt is terminal and closed. This is consumed
development evidence on the already-consumed September 9 Gold11 cohort. It is
not unseen-family validation, calibrated metrology, manufacturing approval or
production release.

## Result

The full paired V8 run finished once (parent wall 9512.7 s, about 2 h 39 min,
versus 6180.0 s for V7). All eleven cases in both arms were retained: 22 durable
case/arm records, 18 returned workflow refusals and 4 CPU-terminated workers
with explicit supervisor refusals. No returned workflow emitted a measurement.

| Metric | Fresh control (V7 scan planes, schedule V1) | Fresh challenger (composite scan planes, schedule V2) | V7 (both arms) |
| --- | ---: | ---: | ---: |
| True positives | 0 | 0 | 0 |
| False positives | 0 | 0 | 0 |
| False negatives | 71 | 71 | 71 |
| Precision | null (no predictions) | null | null |
| Recall | 0 | 0 | 0 |
| Localized F1 | null (undefined) | null | null |
| Minimum-family recall | 0 | 0 | 0 |
| Emitted support lines | 0 | 0 | 0 |
| Refused cases | 11/11 | 11/11 | 11/11 |
| Physical observed counts unavailable | 11/11 | 11/11 | 11/11 |

The unchanged frozen scorer (`GOLD11_CONTACT_GEOMETRIC_DEVELOPMENT_COMPARISON_V2`,
wrapper `GOLD11_SCHEDULE_CONTEXT_BOUND_CONTACT_PROJECTION_V3`, reference
snapshot `d16390b2…2b45dc`, 71 reference lines) returns null precision and null
F1 when there are no predictions. That policy is not replaced after seeing the
result. **There is no numeric new F1.** Paired deltas are zero on every metric.
Zero emitted supports do not mean the scans contain zero bends; the 71 scoring
misses are not 71 measured physical defects. Historical F1 0.4580152672 remains
non-comparable and is not beaten. Both arms changed together (V15 engine,
context V4, surface V4, registration V4, the provisional 1.0 mm pose-admission
gate, the context schedule); results are combined consumed-development changes,
not a mechanism-only speedup or a calibrated accuracy improvement, and the arm
comparison is scan-plane-lane/schedule only, not old-engine versus new-engine.

## Immediate causes (mutually exclusive over all 22 slots)

| Stopping mechanism | V8 slots | V7 slots | V8 cases |
| --- | ---: | ---: | --- |
| Exact raw-extrema coordinate-work limit | 0 | 10 | none: every context stage completed at 25.1M–89.2M coordinate visits |
| Completed finite proposal search, no passing start | 12 | 4 | CG08506, CG64, CG65, CG84, CGEBA, CGF41 (both arms) |
| CPU termination inside registration surface queries | 4 | 1 | CG191, CG4FAF (both arms), killed at the 1860 s CPU limit |
| Scan-plane triangle-membership work limit | 2 | 2 | CGE4B (both arms), unchanged frozen V5 hard maximum 20M |
| Completed residual/context rejection at the declared gate | 1 | 1 | CG4BDD challenger: context p95 upper 1.2009 mm > 1.0 mm |
| Admitted context, zero valid support-cover measurements | 3 | 0 | CG1E0 (both arms), CG4BDD control: all native windows refused |
| Competitive-triangle interval too wide | 0 | 2 | — |
| Near-tie storage limit | 0 | 2 | — |

Two returned outcomes (CG64, both arms) also report INVALID_NATIVE_SOLID; the
V7 native-solid blockers on CG191 and CG4FAF could not be observed because
those workers were killed before returning.

### Per-slot transitions

Wall is the leaf receipt wall; RSS is the sampled leaf-tree peak (not continuous).

| Case | Family | Arm | V7 cause | V8 cause | V8 wall s | V8 peak RSS MiB |
| --- | --- | --- | --- | --- | ---: | ---: |
| CG08506AE97D | 10839000 | C | extrema cap | no passing start | 50.7 | 1573 |
| CG08506AE97D | 10839000 | H | extrema cap | no passing start | 70.1 | 1823 |
| CG191DEC7F81 | 49001000 | C | interval width | CPU kill in refine | 1861.4 | 1351 |
| CG191DEC7F81 | 49001000 | H | interval width | CPU kill in refine | 1861.3 | 1559 |
| CG1E0BBA2933 | 38172000 | C | extrema cap | admitted, no cover measurements | 362.9 | 1530 |
| CG1E0BBA2933 | 38172000 | H | extrema cap | admitted, no cover measurements | 398.6 | 1745 |
| CG4BDD1371B6 | 37361005 | C | residual gate (0.5 mm) | admitted, no cover measurements | 259.4 | 1419 |
| CG4BDD1371B6 | 37361005 | H | near-tie limit | residual gate (1.0 mm) | 293.9 | 1547 |
| CG4FAF273093 | 49024000 | C | CPU kill | CPU kill in refine | 1860.9 | 490 |
| CG4FAF273093 | 49024000 | H | near-tie limit | CPU kill in diagnose | 1860.9 | 514 |
| CG64C412CC73 | 40500000 | C | no passing start | no passing start (identical funnel) | 30.1 | 1007 |
| CG64C412CC73 | 40500000 | H | no passing start | no passing start (identical funnel) | 39.0 | 1077 |
| CG65CD575B1C | 38172000 | C | no passing start | no passing start (identical funnel) | 38.3 | 1610 |
| CG65CD575B1C | 38172000 | H | no passing start | no passing start (identical funnel) | 56.0 | 1788 |
| CG84C8FA12D2 | 10839000 | C | extrema cap | no passing start | 52.1 | 1689 |
| CG84C8FA12D2 | 10839000 | H | extrema cap | no passing start | 74.1 | 1867 |
| CGE4B0549E41 | 10839000 | C | membership cap | membership cap | 13.6 | 1603 |
| CGE4B0549E41 | 10839000 | H | membership cap | membership cap | 13.6 | 1674 |
| CGEBA9A869AC | 38172000 | C | extrema cap | no passing start | 38.1 | 1318 |
| CGEBA9A869AC | 38172000 | H | extrema cap | no passing start | 53.1 | 1402 |
| CGF41F49066A | 38172000 | C | extrema cap | no passing start | 42.8 | 1468 |
| CGF41F49066A | 38172000 | H | extrema cap | no passing start | 61.0 | 1618 |

Family false negatives (both arms): 10839000 18, 37361005 4, 38172000 34,
40500000 7, 49001000 5, 49024000 3.

## What changed stage by stage

### Raw-extrema context stage: the V7 blocker is gone

Every case that reached the two-plane context completed it. The five scans that
V7 refused at exactly 256,000,000 coordinate visits (after roughly 26.6M–27.1M
evaluated query rows) completed under the kd-partition index V3 with the
following counters (identical in both arms):

| Case | Coordinate visits | Charged queries | Evaluated query rows | Signed families |
| --- | ---: | ---: | ---: | ---: |
| CG08506AE97D | 89,178,513 | 219 | 7,005,730 | 184/184 |
| CG1E0BBA2933 | 70,439,115 | 492 | 5,216,955 | 320/320 |
| CG84C8FA12D2 | 79,883,302 | 174 | 5,811,800 | 192/192 |
| CGEBA9A869AC | 52,584,781 | 471 | 3,659,238 | 304/304 |
| CGF41F49066A | 27,774,069 | 135 | 439,367 | 80/80 |

All completed contexts used fewer coordinate visits than the old 256M cap, so
the raised 512M cap was not what admitted them; this is an observation on these
five scans under the V3 index, not a general efficacy claim (the synthetic
small-batch regression recorded in the delta review is retained).

### Initialization funnel: the exposed blocker

Removing the context refusal exposed the next stage. For six scans (12 slots)
the complete finite proposal search rejects every generated start; CG64 and
CG65 reproduce their V7 funnel counters exactly.

| Case (both arms) | Proposed hypotheses | Search-stage rejections | Assigned supports box-screened | Bound-excluded supports | Completed finite calls / rows | Passing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CG08506AE97D | 4,500 | 3,492 | 12,356 | 12,159 | 197 / 20,196 | 0 |
| CG64C412CC73 | 4,454 | 3,462 | 12,358 | 12,263 | 95 / 10,317 | 0 |
| CG65CD575B1C | 2,045 | 6,059 | 5,935 | 5,930 | 5 / 533 | 0 |
| CG84C8FA12D2 | 4,522 | 3,326 | 12,342 | 12,127 | 215 / 23,418 | 0 |
| CGEBA9A869AC | 4,862 | 2,976 | 12,572 | 12,440 | 132 / 14,242 | 0 |
| CGF41F49066A | 4,272 | 3,776 | 12,288 | 12,118 | 170 / 19,292 | 0 |
| CG1E0BBA2933 | 4,876 | 2,912 | 12,504 | 12,332 | 172 / 18,442 | 1 |
| CG4BDD1371B6 (C / H) | 4,938 / 4,928 | 2,656 / 2,720 | 12,532 / 12,576 | 12,264 / 12,366 | 268 / 28,879 ; 210 / 25,164 | 1 / 1 |

The dominant hypothesis rejections are `NOMINAL_WAHBA_WITNESS_OUTSIDE_DECLARED_GATE`
(search stage) and `ABSENT_EMPTY_HEURISTIC_INTERVAL`; the few hundred finite
face queries per case complete, and individual supports can pass with coverage
1.0 while the whole hypothesis is rejected as
`ASSIGNED_FINITE_FACE_PREFILTER_REJECTED` because every assigned support must
pass. This is complete evaluation of the declared finite triplet/pair proposal
set (search not exhaustive, pair-lane starvation flagged possible), not a proof
that no physical pose exists.

### Registration: two scans now exhaust CPU inside surface queries

CG191 and CG4FAF reached whole-part refinement in both arms and were killed at
the 1860 s CPU limit (exit -24, supervisor refusal `INFERENCE_WORKER_FAILED`,
no RESULT.json, final geometry unknown). The 62 main-thread stage samples per
worker place more than 80% of samples in `_paired_candidates`/`leaf`/`walk` of
`cad_guided_triangle_surface_v4.py` called from registration `_refine` (three
slots) or `_diagnose` (CG4FAF challenger, which had finished refinement). In
V7 these slots failed earlier and faster (interval width at 1355/530 s; near-tie
at 1316 s; one CPU kill). The 256-slot tie cap and the sound vertex pruning let
these queries continue, but the competitive-candidate enumeration on these two
geometries is too expensive for the frozen CPU budget. High-water RSS at the
last sample: 1.42/1.64 GB (CG191) and 0.51/0.54 GB (CG4FAF). No internal stage
completion is inferred for killed workers.

### Pose admission: what the provisional 1.0 mm gate did

| Slot | Gate metric: inlier-plane p95 (mm) | Context support p95 upper (mm) | Coverage lower | Reverse sample coverage | Under V7 gate 0.5 mm | Under V8 gate 1.0 mm |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| CG1E0 control and challenger | 0.8480 | 0.9008 | 0.9992 | 0.9811 | would be rejected | SUPPORTED_CONTEXT |
| CG4BDD control | 0.6659 | 0.6956 | 0.9999 | 0.9908 | rejected in V7 (same numerics) | SUPPORTED_CONTEXT |
| CG4BDD challenger | 1.1959 | 1.2009 | 0.9997 | 0.9832 | rejected | rejected (SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE) |

The three admitted contexts were scheduled (1 supported, 1 scheduled, cap 6
from the fixed measurement preflight budgets) and reached native features
(PARTIAL; 13, 13 and 17 transition windows) and the exact raw support-cover
measurement. The measurement stage cropped 9.60M/9.60M/9.88M rows and refused
every window: 39 windows `FLANGE_NORMAL_WEAK_RELATIVE_TO_RESIDUAL_POLICY` and 4
windows `FLANGE_NOT_PLANAR_AT_DECLARED_RESIDUAL`, so zero conditional support
records, zero cover components and zero predictions. The relaxed gate therefore
produced no prediction, true or false. What was observed is the local refusal
of every window by the unchanged sample-model gates (weak second spread or
out-of-plane ratio; max residual above 0.05 mm). The global proximity p95 of the
admitted pose is NOT the local flange residual: the centered SVD is invariant to
a rigid pose for fixed rows, and the pose enters only through crop membership.
No false-pose consequence was observed, and this document does not attribute
the window refusals to the admitted residuals (an earlier draft did; that claim
was unsupported and is withdrawn). Competing causes remain open: mixed skins
inside the 3 mm crop margin, narrow windows, noise and wrong-context crops; the
V9 diagnosis records the evidence for each and the successor must publish the
discriminating per-window diagnostics (N, spectrum, selected-row/extent
provenance, actual residuals, both flange sides, both weak-gate branches) with
the final gates unchanged. Uncertainty remains uncalibrated; conformance is
INDETERMINATE everywhere.

Metric note: the registration gate compares the candidate's inlier-plane p95
(`inlier_plane_p95_mm`; CG4BDD control 0.6659 mm, the value the frozen
`CONFIG_DELTA_FROM_V7` cites as 0.666 mm) while the context support record
carries the finite-distance p95 upper (0.6956 mm, the value the V7 result
document cites). Both are V7 numerics reproduced exactly by V8; the earlier
draft of this document wrongly called this a discrepancy.

### Unchanged blockers

CGE4B fails in both arms at `MAX_TRIANGLE_MEMBERSHIPS_EXCEEDED` in the frozen
noisy-plane V7 lane (the composite lane depends on it), as predicted in the
delta design. CG64 (both arms) still reports INVALID_NATIVE_SOLID alongside its
prefilter negative.

## Execution and custody evidence

- Sole invocation, after the independent PRE (GREEN, bundle
  `43173e8a…a326`) and an in-launcher fresh no-overlap process check, at
  2026-09-12 23:52:07 UTC: `OMP_NUM_THREADS=1 .venv/bin/python -I -S -B -u
  scripts/run_gold11_mesh_development_v8.py --execute`, detached in its own
  session; parent process disappearance observed by ps at 02:30:42 UTC (exit
  status not observed, see below). This is an execution record,
  not a retry command: the attempt is terminal and must not be rerun, resumed,
  overwritten or rescored.
- Final printed status `COMPUTED_PENDING_INDEPENDENT_POST_AUDIT`, 22 durable
  terminal records, final campaign inventory
  `dbeb824d1929e6f3b638cfc3cc1f8ed13899fc5a5da39a62d5fbc7a2912445bb` (283
  entries, 115 scientific manifest entries). **The parent's process exit
  status is unobserved**: the detached launch had no waiting parent, so no
  shell exit code was captured. FINAL_GATE.json present and
  FINALIZATION_FAILED.json absent are consistent with a zero return from
  `execute()`, but they do not prove the process exit status, and the runner's
  own release condition (`OBSERVED_PARENT_EXIT_ZERO_FINAL_GATE_NO_VETO_AND_
  EXACT_INDEPENDENT_POST_AUDIT`) is therefore not met by observation. This is
  a retained evidence limitation of this attempt; it cannot be repaired by a
  rerun, and the POST is asked to record it rather than infer exit 0.
- COMPLETED.json: failures [], final source audit unchanged (74 frozen sources
  equal current and frozen bytes), parent wall 9512.744 s, parent lifetime peak
  RSS 1,557,233,664 B. Externally sampled tree peak 2.41 GiB (20 s ps polling,
  not a continuous guarantee). Scorer worker exit 0, sampled peak 87.6 MB.
- Scoring happened only after INFERENCE_TERMINAL (22 records, parent wall
  9392.1 s at that boundary); SELECTED_REFERENCES.json is produced by the
  scoring phase. Gold stayed outside inference.
- No attempt or cache artifact was read during the run; monitoring was ps-only.
  The coordinator private cache is retained opaque outside the attempt.

## Independent acceptance

- Physical PRE (before execution): `2026-09-13_gold11_mesh_development_v8_physical_pre_run_audit.md`,
  19,219 B, SHA256 a0faba1a32378f84dfe176e9fb1bb839e1c60bc8ec487b84f53b5b158abb2352;
  `PRE_RUN_AUDIT.json` 20,736 B, SHA256 41fd61fc6157e44823477e04da034d4b378307055cb325e0bcb19f1d7c7d6554.
- Physical POST (after termination, non-author AI auditor, read-only, written
  to job scratch and copied verbatim): `2026-09-13_gold11_mesh_development_v8_post_run_audit.md`,
  44,587 B, SHA256 894dfb8a606a02825e4732883449352220b086c5897e76c5ba953a218fca10c1.
  Decision: ACCEPT execution custody and the complete negative scoring ledger,
  as an engineering audit only, with the explicit limits that the parent exit
  status is unobserved (release condition not met by observation, not
  repairable by rerun), that no numeric F1 exists (both arms TP 0 / FP 0 /
  FN 71, F1 null), and that nothing here is manufacturing approval, calibrated
  metrology or generalization evidence. The POST reproduced the final inventory
  digest, all control digests (STARTED/VERIFIED/PRE/SCORED bundle digests,
  FINAL_GATE completed digest), 74/74 sources three ways, all 115 declared
  files and 281 inventory fingerprints, the INFERENCE_TERMINAL outcomes digest
  from the 22 SCORING_INPUT files and the scorer request digest; scoring ran
  59.4 s after INFERENCE_TERMINAL with no later case-file change; no retry,
  resumed start, relaxed limit, edited source or veto marker. It notes that the
  18 returned workers' CPU rose from 925.75 s (V7) to 1850.33 s (not a
  speedup) and that all 74 frozen sources exist in the working tree only as
  untracked git files (custody rests on the snapshot and archive).
- POST addendum (same non-author auditor, separate file, original preserved):
  `2026-09-13_gold11_mesh_development_v8_post_run_audit_addendum.md`, 6,953 B,
  SHA256 39b2547c350ca93b81ba6cfc8476d08d03ae47c3c75a892283eda0e0af981f8d.
  It corrects two prose sentences of the original that said all ten former V7
  extrema-refusal slots became no-passing-start outcomes: eight did (CG08506,
  CG84, CGEBA, CGF41, both arms) and the two CG1E0 slots reached measurement
  under the 1.0 mm gate and were refused there. Every count, table row, score
  and binding in the original was re-derived unchanged; the decision (ACCEPT
  with limits) is unchanged.

## Next justified engineering step

The context-capacity blocker is closed on this cohort; the run did not fail on
memory or wall. Three separate blockers now stand between the engine and any
measurement, in order of how many slots they stop:

1. **Finite proposal coverage (12 slots).** Thousands of triplet/pair starts are
   generated and none survives the search gate plus the all-supports-must-pass
   finite prefilter. The next mechanism must widen or re-rank the proposal set
   or the assignment rule without tuning to case identities: e.g. a separately
   versioned proposal lane whose witness gate and support assignment are
   declared from CAD/scan geometry only, with exact accounting and retained
   negatives. CG64/CG65 are unchanged controls for it.
2. **Surface-query CPU on CG191/CG4FAF (4 slots).** The V4 competitive-candidate
   leaf pairing dominates CPU. A successor needs a cheaper exact certificate
   (e.g. tighter pass-1 bounds or batched leaf pairing) proven on a fixed
   mechanism matrix with full build-plus-query CPU, not a larger CPU limit for
   a started attempt.
3. **Measurement window refusals (3 slots, 43 windows).** Every window failed
   the local sample-model gates; the cause is not established by this run and
   is not the admitted global p95. The successor must publish discriminating
   per-window diagnostics (both flange sides, both weak-gate branches, N,
   singular spectrum, selected-row/extent provenance, actual residuals) and may
   add a declared geometric row-selection mechanism only with that evidence
   alongside it; the sample-model gates stay unchanged and the provisional
   1.0 mm admission gate must not be raised to manufacture output.

The membership-cap (CGE4B) and INVALID_NATIVE_SOLID blockers remain explicit
and deferred. No tolerances, datums or nominal CAD measurements will be
invented to make outputs appear. Any successor needs a documented engineering
reason, a separately versioned frozen candidate and an independent PRE before
another consumed-Gold11 execution; V8, V7 and every earlier attempt remain
terminal and unchanged.
