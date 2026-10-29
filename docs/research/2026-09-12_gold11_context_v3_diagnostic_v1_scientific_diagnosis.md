# Context V3 six-slot diagnostic: bounded scientific diagnosis

Date: 2026-09-12. This is scientific interpretation of the released terminal derived artifacts, not the independent custody POST or a new release. No raw inputs, Gold, references, private caches, other attempts, geometry execution, target imports, or tests were accessed/performed for this diagnosis. No candidate source was changed.

## Evidence boundary and identity

Read the complete frozen protocol first: `gold11_context_v3_diagnostic_v1_attempt/source/docs/research/2026-09-12_gold11_context_v3_diagnostic_v1_protocol.md`, 12,845 bytes, SHA256 `da2db6127916ec2b714fa03c3f04cc901c0ff1263ed35db1b1c8d39d87822ec3`. Case reads were confined to the three declared cases, control then challenger, and their declared COMPLETED, OUTPUT, RESULT and SCORING_INPUT.json artifacts. Receipt/projection, log and result limits were respectively 8 MiB, 1 MiB and 128 MiB; all observed sizes were below those limits before content processing.

Root `COMPLETED.json`: 79,461 bytes, SHA256 `375c0af4d45fdc5e457e85325f86a8a3bd5409024931698f619dc99a1951566e`. `DIAGNOSTIC_EVIDENCE.json`: 35,078 bytes, file SHA256 `64c28d839b6a9976b23ce32a9b7bf9f3490d3866a330d252c63aee1d2b1571d4`. Its canonical compact, sorted-key JSON SHA256 is `16607180115cbc36357f4efab721db93503dd9d565215e6c70e95573a9e94022`, exactly root COMPLETED's `evidence_sha256`; the canonical digest is intentionally different from the formatted file digest. Canonicalization used only stdlib JSON/hash processing, not target code.

All 23 declared case files independently matched their evidence-declared byte counts and SHA256 values. The five terminal log RESULT hashes also match retained RESULT hashes. Six leaf receipts have `source_binding_valid=true`, and the root receipt records six durable slots and no failures, with status `DIAGNOSTIC_SIX_SLOTS_UNSCORED_PENDING_INDEPENDENT_POST`. These are retained claims plus the bounded checks above, not an independent re-audit of every custody condition. Root separately supplied post-exit external writer-quiescence observation; the custody reviewer owns formal POST.

## Full six-slot negative ledger

Wall time is the leaf supervisor's `wall_seconds`; CPU is the returned `INFERENCE_TERMINAL` log value except where explicitly unavailable. Rounding here does not replace the exact retained receipt.

| Case / arm | Exit | Wall s / CPU s | Returned scientific outcome |
|---|---:|---:|---|
| CG08506AE97D / control | 0 | 36.164 / 35.991 | UNAVAILABLE, WHOLE_PART_REGISTRATION; initialization refused at the context extrema-coordinate work cap. |
| CG08506AE97D / challenger | 0 | 55.509 / 55.343 | Same nested refusal and exactly the same context work counters. |
| CG4BDD1371B6 / control | 0 | 228.582 / 228.232 | One candidate completely evaluated; no geometrically supported measurement context because residual gates fail. |
| CG4BDD1371B6 / challenger | 0 | 125.956 / 125.679 | One candidate attempted; finite-surface near-tie capacity refusal aborts candidate evaluation. |
| CG4FAF273093 / control | -24 | 1863.759 / final unknown | No returned RESULT; supervisor `INFERENCE_WORKER_FAILED`. Last sample CPU 1859.401 s supports CPU-limit attribution, not a returned geometry diagnosis. |
| CG4FAF273093 / challenger | 0 | 1353.789 / 1352.582 | Eleven candidates retained, four attempted: three residual-rejected, fourth near-tie refusal, seven explicitly not attempted after fatal failure. |

Five ordinary exits produced five UNAVAILABLE registration results, with no predictions and no measurement result. None is a valid-geometry success. The sixth slot is a resource-terminated unknown, not a fabricated registration rejection or zero-bend observation. All six slots remain in the diagnostic denominator. There is no F1, reference comparison or scoring here.

## CG08506AE97D: exact context refusal, both arms

The reason chain is `NOISY_PLANE_INITIALIZATION_UNAVAILABLE` → `TWO_PLANE_CONTEXT_UNAVAILABLE` → `BUDGET_EXTREMA_COORDINATE_VISITS`. The actual attempted counter is **256,001,863** against the requested **256,000,000** cap. It is not the older 64-million parent scan-row budget: retained parent context scan-row visits are 29,171,478 and index visits 35,881,868, both below 64 million. The refused helper clears usable output and has no completed budgets publication; its retained reason/work ledger plus the bound requested configuration establish which limit fired.

Both arms retain exactly: 11 completed extrema indexes, 819,364 indexed support-row memberships, 127 charged queries, 26,714,627 successfully evaluated query rows, 237,328 extrema node visits, 104,408 leaves evaluated and 12,169 nodes pruned. Registry hits are 206; support aliases 10; alias-comparison rows zero. Projection cache hits/misses/entries are 264/169/168. Indexed rows are cumulative memberships across indexes, not a claim of that many distinct raw points. Query count is charged on entry and does not mean 127 completed queries.

Source-supported accounting explains the exact overrun: `cad_guided_raw_extrema_index_v1.py:159` increments a work counter before checking its cap; `_build` at line 177 charges 12N + N + 6N = 19N coordinate visits, and `_evaluate` at line 250 charges 9N before evaluating/incrementing successful query rows. Thus:

`19 × 819,364 + 9 × 26,714,627 = 255,999,559`.

Only 441 units remain. The next 256-row leaf charge is 2,304 units, yielding exactly 256,001,863, 1,863 above the cap. This is a source-and-counter inference identifying the refused precharge, **not** evidence that the over-budget coordinate arithmetic actually executed. The cap is explicit in `cad_guided_two_plane_context_v3.py:38–43` and the unchanged V6 request at line 241.

The helper planned 184 signed families / 1,472 slots, attempted 73 families, completed 72 / 576 slots (472 proposed, 104 absent). Initialization assigned-support, box-screen and finite-query counters remain zero; registration candidate count is zero. Therefore this refusal is upstream of assigned-support screening and iterative registration. Extrema accounted working peak is 58,952,984 bytes, parent context accounted working peak 118,611,454 bytes; these are payload-accounting counters, not RSS. The terminal logs report process high-water RSS of 1,583,316,992 and 1,799,127,040 bytes respectively. A coordinate work budget, not an observed RSS cap, caused the returned refusal.

## Registration: CG4BDD1371B6 and CG4FAF273093

The actual chain is V14 `inspect_cad_guided_mesh_v14:400` → V13 `register_measurement_contexts_v13:584` → `register_whole_part_planes_v13`, with coarse `_refine` at line 511, fine `_refine` at 513, `_diagnose` at 516 and `_context_support` at 518. `_Queries.__call__` in global registration V3:209–235 invokes the finite-triangle query and retains valid attempted work even on whole-query refusal. Its cumulative registration counters are distinct from each individual surface-query cap.

| Returned arm | Initialization support calls / rows | Bound-excluded calls / rows | Finite calls / rows | Registration candidates attempted / retained | Registration calls / rows | Registration work units |
|---|---:|---:|---:|---:|---:|---:|
| CG4BDD control | 12,532 / 1,614,486 | 12,264 / 1,585,607 | 268 / 28,879 | 1 / 1, complete | 63 / 203,331 | 122,818,771 |
| CG4BDD challenger | 12,576 / 1,706,024 | 12,366 / 1,680,860 | 210 / 25,164 | 1 / 1, incomplete | 31 / 36,553 | 21,617,158 |
| CG4FAF challenger | 3,566 / 388,355 | 3,434 / 374,041 | 132 / 14,314 | 4 / 11, incomplete | 200 / 671,379 | 649,562,167 |

These three returned initializers and context helpers are COMPLETE. They do not reproduce the CG085 coordinate refusal. Context coordinate visits are respectively 89,838,107; 89,716,901; and 5,596,501. The finite prefilter is materially rejection-active, but the remaining iterative correspondence workload is separate. For example CG4FAF challenger retains 140,727,204 certificate pairs, 115,212,967 node visits, 35,181,801 triangle pairs and 2,305,106 normal pairs across registration calls. These cumulative counts exceeding some *per-call* maxima do not establish a budget violation; `_Queries` also applies the remaining cumulative registration allowance.

CG4BDD control's single candidate completes 30 coarse and 30 fine iterations, both `ITERATION_LIMIT` (not a convergence certificate). Its scan coverage lower bound is 0.9882774656 and reverse CAD coverage 0.9907531738, yet inlier surface p95 upper is 0.665897241 mm, above the declared 0.5 mm residual gate. Conditional-context proximity covers 20,642 / 20,644 rows (lower bound 0.9999031196), but its p95 interval is [0.6956471254448494, 0.6956471254551617] mm, also above 0.5. Candidate reason: `SURFACE_RESIDUAL_OUTSIDE_DECLARED_ALGORITHM_GATE`; context reason: `CONTEXT_PROXIMITY_RESIDUAL_OUTSIDE_GATE`; final registration: `NO_GEOMETRICALLY_SUPPORTED_MEASUREMENT_CONTEXTS`. High coverage does not satisfy the independent residual condition (`global_registration_v3.py:458`, initialization V12:1074–1097).

CG4BDD challenger instead ends with `FINITE_SURFACE_QUERY_UNAVAILABLE:NEAR_TIE_BUDGET_EXCEEDED`. Its completed coarse/fine fields are absent, because V13 publishes them only after the entire per-candidate sequence. Retained row counts provide a narrower inference: 31 queries / 36,553 rows = 30 × 1,079 coarse rows + 1 × 4,183 fine rows. Together with a last sample inside V13:513, this supports failure on the first fine-stage query after 30 coarse queries; it is not a completed fine-stage result.

Both near-tie-refused arms declare `max_ties_per_query=32`. At triangle-surface V3:482–497, the second, fixed-upper-bound pass retains competitive triangle intervals and refuses before inserting another triangle when a query-point slot count has reached 32. This is capacity for competing represented-triangle possibilities per query point, not proof of 33 physically equal distances, a specific bad raw row, or a normal-pair budget failure. No such row/triangle identity is recovered from these compact aggregate reports.

CG4FAF challenger has 11 exact passing start groups. Candidate indices 0–2 each complete 30 coarse plus 30 fine iterations, both iteration-limited. Their scan/CAD coverages and surface p95 upper values are respectively (0.821269 / 0.829956 / 1.550065 mm), (0.821398 / 0.830109 / 1.549431 mm), and (0.530498 / 0.586426 / 1.450295 mm). All exceed the 0.5 mm residual gate; candidate 2 also lacks sufficient support. Their conditional-context p95 upper values are 1.622713, 1.622852 and 1.587352 mm; all contexts are rejected. Candidate 3 receives the near-tie fatal reason; candidates 4–10 retain `NOT_ATTEMPTED_AFTER_FATAL_FAILURE`. V13:534–558 clears selected contexts and marks candidate evaluation incomplete after the fatal failure. The separate returned native-feature ledger also says `UNAVAILABLE / INVALID_NATIVE_SOLID`; that is not the immediate registration refusal and was not experimentally investigated.

## CG4FAF control: CPU evidence, not an unseen result

The leaf receipt has exit -24, `budget_stop=null`, supervisor refusal `INFERENCE_WORKER_FAILED`, and retention status `NO_RETURNED_RESULT_INTERNAL_STAGES_UNKNOWN`. Its reported `ValueError` is a supervisor-side failure type, not a recovered native-engine traceback. The signed exit and last CPU observation are consistent with the declared Darwin 1,860-second worker CPU limit (V6:1427 sets RLIMIT_CPU); the receipt does not contain a final CPU measurement or an explicit parent CPU-budget stop.

All six logs parse as complete JSON lines within the 1 MiB limit. This control log contains the workflow-call-begin record plus 62 sampler records (indices 0–61), no sampler finish receipt and no INFERENCE_TERMINAL. Every sample is in finite-triangle registration work. V13 call-site counts: coarse line 511, 21 samples; fine line 513, 35; candidate diagnosis line 516, four; context support line 518, two. Across those samples, 42 show the first closest-surface pass at line 524 and 20 show the fixed-upper/tie pass at line 543.

The last record, OUTPUT line 63, has elapsed 1,860.705760625191 s, CPU 1,859.40107 s and process high-water RSS 504,414,208 bytes. Its innermost reported frame is `_paired_candidates`, triangle-surface V3:296, through `leaf:459`, `walk:522`, `query_closest_surface:524`, `_Queries.__call__:222`, `_refine:353`, V13:513, V14:400, V6 wrapper:1399 and worker:1439. One additional named frame is omitted by the 12-frame limit.

This supports a CPU-resource limitation while repeatedly computing finite-triangle correspondences/refinement, not an initialization-cap failure or a measurement-stage hang. Earlier diagnosis/context-support samples prove that some candidate work reached those call sites, but do not supply final candidate counts, poses, acceptance, or exact unseen accumulated counters. No infinite loop is established: the source has bounded candidates, iterations, traversals and work budgets, whose repeated permitted work can nevertheless consume the CPU envelope.

Sampler V1 observes allowlisted Python code locations, innermost first, with up to 12 named frames / 64 walked frames at nominal 30-second intervals. It does not inspect locals, attribute time continuously, identify the exact native instruction, or certify stage completion; scheduling/GIL delays and sampling gaps remain. The 62 samples are not an exact CPU-time allocation among stages. The last sample is not necessarily the interruption point. Absence of a finish record is not independent evidence that an engine phase was never reached.

## Scope and next-step implications

All nine source files cited above were hashed and match the corresponding root COMPLETED post-quiescence source-audit declarations, including V6 SHA256 `7035541386dee007425cb728a6c3a854f01fa4a0ea87aca44d3849725f2e3c80`, V13 `79ee4de537f1dddcc1dee40d33b25fe3fe8fb667bcd4e30ecc0a811ebe2d989f`, V14 `76378b93c44c046b773452b3d8cad400cb237ff8bc7039e258a90305bbf6c749`, context V3 `9077f434aa7baf0ce5b63c620407b091258bb41e2141bbe41465ca4074ed543d`, and sampler `80316d0026bcc2fabcf1d8861d010d5f434f720c303f5969384cc45f7809b6b0`. Source links/lines here describe those unchanged files, not a newly executed reconstruction.

The five results return during WHOLE_PART_REGISTRATION, before V14's registration acceptance and measurement transitions at lines 414 and 458. Their null measurement fields therefore do not assess downstream measurement capacity. The control CPU slot has only sampled phase evidence, not a returned phase ledger. A separately noted source-only measurement crop-work limit is not an observed cause in any of these six slots.

These observations do not justify silently raising limits, changing residual/tie tolerances, accepting partial candidate work, or tailoring future case selection. Both arms use context V3; this is the explicitly declared new comparator, not an unchanged historical control. There is no old-attempt comparison or causal claim that a previous failure was fixed. The frozen protocol permits scientific negatives/resource terminations as diagnostic outcomes when custody remains interpretable. Subject to the existing independent POST/release gates, this diagnosis introduces no all-positive prerequisite, new broad gate, or delay requirement for the planned full-eleven/both-arm/unchanged-71 scoring successor.
