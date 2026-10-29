# Sherman QC F1 — Post-V55 terminal next-action packet

You are the research professor selecting exactly one next bounded action for the active Sherman QC F1 generalization objective.

## Required ruling

Return exactly one of:

- `RULING: EXECUTE_ONE` followed by one fully specified, preregisterable intervention with a fresh name and no V55 rerun or rescue; or
- `RULING: OPEN_CAPABILITY` followed by the single smallest scientifically necessary capability opening and one bounded execution that uses it; or
- `RULING: STOP_RESEARCH` only if the evidence establishes that neither a new source-only mechanism nor a minimal capability opening can falsifiably advance the objective.

Prioritize one fast falsifiable execution over further paper-only review. We want to hit the next wall through action, score once if a complete output freezes, and inspect the geometry visually—not optimize against locked answers. The selected intervention will be implemented once, tested only on source-free fixtures before data execution, executed once on the sealed non-confirmatory synthetic lane with primary/replay determinism if it reaches replay, scored once only after the output freezes, and manually audited at native resolution. No threshold rescue, rerun, replacement input, or post-hoc variant is allowed.

## Frozen objective

Develop a trustworthy zero-shot bend-inspection system for known and near-neighbor sheet-metal families. Final enterprise gates remain at least 90% exact-count accuracy, 90% bend-line precision, 90% recall, 90% F1, 99% render-admissible marker accuracy, 100% deterministic replay, peak RSS below 16 GiB, P95 latency no more than 300 seconds, at least 99.5% job success, and no major family below 80% recall. The 30-scan/15-family synthetic lane described here is engineering evidence only and cannot establish physical-scan or paper claims.

## Retained baseline: V52

`V52_CYLINDRICAL_CONSENSUS_RECOVERY_V1` remains frozen and retained:

- 30 scans / 15 families;
- TP 52, FP 27, FN 48;
- precision 0.6582278481;
- recall 0.52;
- F1 0.5810055866;
- exact-count accuracy 6/30;
- minimum family recall 0;
- 30/30 deterministic replay;
- 22 append-only additions, all 22 matched truth after the sealed score;
- native-resolution review confirmed the added axes visually land on physical folds.

The frozen union pool has a 91/100 candidate ceiling and 22/30 scans with complete candidate birth. Nine truth bends are absent from the whole pool. V52's 27 false positives also prove that an append-only route cannot reach the 90% precision gate. A successful next architecture therefore needs safe pruning/replacement as well as recovery, or a carefully justified capability opening.

The six exact V52 cases are S005, S006, S007, S013, S026, and S027. Known difficult visual cases include S011, S016, S024, and S029.

## Retired V53 and V54

V53 ALBC required unrepresented long candidates to have same-bank agreement between scan-mesh adjacency and plane-lattice routes. It added no lines because the only cross-route cluster had insufficient common span. V53 is terminal and unavailable for relaxation or rescue.

V54 ENTC used frozen support banks to assign persistent normal-turn mass exclusively along candidates. It terminally stopped at S013 on `ABSTAIN_V52_PROVENANCE_MAPPING`. Native-resolution review of its two partial additions, S010 and S012, rejected both: the axes lay on flat panel interiors while the real rounded corners were transversely displaced. V54 is terminal and unavailable for correction or rescue. Its evidence established that strong longitudinal aggregate support does not prove transverse axis localization.

## V55 hypothesis and authorization

The post-V54 ruling selected `V55_MESH_SECTION_TURN_LOCUS_RESELECTION_V1` (`MSTLR_V1`) with precommit `V55_MSTLR_MECHANISM_LOCKED` and terminal token `V55_MSTLR_TERMINAL_RETIRE`.

V55 was a full reselection architecture over frozen V52 predictions and the frozen union pool. For each explicit candidate line, it sampled nine longitudinal stations, intersected the source mesh with axis-normal planes, traced deterministic mesh-section polylines, measured direct tangent turn loci, distinguished centered from offset folds, pruned V52 candidates only under a fail-closed negative certificate, proposed pool additions only under a strict positive certificate, and applied one deterministic conflict pass. It used no V53/V54 behavior, no truth, labels, CAD, expected counts, identity, geometry refit, or new candidate birth.

The professor required the entire frozen candidate population to parse as valid lines. A malformed frozen line was a scan-level abstention and any scan-level abstention immediately terminally retired the architecture. V55 could not skip rows or infer missing endpoints after execution began.

## V55 preflight evidence

Implementation tests passed before execution: `5 passed in 2.02s`.

All ten professor-specified source-free fixtures passed:

1. centered sharp 90-degree fold;
2. centered rounded 60-degree fold;
3. single plane;
4. fold offset by 0.40 radius;
5. double fold at plus/minus 0.40 radius;
6. distributed cylinder;
7. open boundary;
8. diagonal cross-panel axis versus true fold;
9. two parallel folds separated by 0.08 bounding-box diagonal;
10. canonical serialization and non-finite rejection.

The frozen implementation and artifact hashes are recorded in `docs/research/F1_SYNTHETIC_LIDAR_STRESS_V2_V55_MSTLR_TERMINAL_CYCLE_20260815.md`. Primary and replay source/V52/pool files were byte-identical for all 30 cases. The largest pool had 755 rows, below the 2,048-row cap. No source points, vertices, faces, memberships, or bitsets were serialized.

## V55 one-shot terminal execution evidence

The one authorized execution was consumed exactly once. It stopped before completing S001:

- worker reason: `ABSTAIN_MALFORMED_FROZEN_LINE`;
- terminal reason: `PRIMARY_WORKER_FAILURE`;
- completed records: none;
- terminal token: `V55_MSTLR_TERMINAL_RETIRE`;
- peak RSS: 338,378,752 bytes, safely below 16 GiB;
- truth never opened;
- no replay, score, changed output, visual artifact, or retention decision;
- no rerun, replacement input, threshold change, or code rescue.

Exact terminal artifacts:

- `output/research/f1-synthetic-lidar-stress-v2-20260814/v55-mesh-section-turn-locus-reselection-v1/primary/worker_failure.json`
- `output/research/f1-synthetic-lidar-stress-v2-20260814/v55-mesh-section-turn-locus-reselection-v1/terminal_failure.json`

Because no case completed, the visual gate was inapplicable rather than bypassed. There is no V55 geometric result to accept or reject.

## Exact failure anatomy

The S001 frozen pool has 17 rows. V55 accepted a row only if it had either:

- `line_start` and `line_end`; or
- `span_endpoints` of length two.

Ordinal 0 satisfied the rule. Ordinal 1 was the first failure:

```text
ordinal: 1
candidate_id: bend_hypothesis::B1
candidate_source: bend_hypothesis
available geometry: anchor, axis_direction, span_mm
missing geometry: line_start/line_end and span_endpoints
```

Ordinal 2 independently had the same mismatch:

```text
ordinal: 2
candidate_id: control_support_strip::S1
candidate_source: control_support_strip
available geometry: anchor, axis_direction, span_mm
missing geometry: line_start/line_end and span_endpoints
```

Most later S001 rows do carry explicit `span_endpoints`. Thus the frozen pool is a heterogeneous proposal table, not a homogeneous endpoint-line table. Candidate count and byte integrity did not imply endpoint-schema totality.

An axis segment could be deterministically materialized from `anchor`, normalized `axis_direction`, and `span_mm`, for example by symmetric endpoints at anchor plus/minus half-span along a canonicalized direction. But V55 did not authorize that mapping. Retrofitting it and rerunning would be post-hoc rescue and is forbidden. Restricting the population after failure to endpoint-bearing rows would likewise be a new population definition, not a V55 correction.

## What V55 did and did not falsify

V55 falsified this implementation assumption:

> every row in the frozen union pool already carries the explicit endpoint geometry required by a line-reselection architecture.

V55 did **not** test, and therefore did not falsify, the mesh-section turn-locus centering hypothesis. No candidate reached that measurement stage on a completed scan. Do not treat this terminal as evidence for or against MSTLR's geometric selectivity.

## Decision that now matters

Select the single most scientifically informative next action without rescuing V55. Plausible routes include, but are not limited to:

1. a fresh architecture whose eligible population is preregistered as only frozen rows with explicit endpoint geometry;
2. a fresh architecture with a total, deterministic schema-normalization layer that materializes explicit lines from every supported frozen geometry schema before geometric evaluation;
3. a fresh source-derived candidate-birth route producing a homogeneous line table; or
4. the smallest capability opening if the present source-only observables or frozen proposals are no longer sufficient.

Please choose exactly one route. Do not simply say to patch V55. If you choose schema normalization, specify which source kinds/field combinations are supported, the exact canonical endpoint formula, treatment of invalid/ambiguous direction/span/anchor values, whether unsupported rows abstain at candidate or scan level, duplicate handling, canonical ordering, and why the mapping does not introduce hidden post-hoc tuning. If you choose endpoint-only eligibility, justify why excluding non-endpoint rows is scientifically acceptable and quantify the required pre-execution population audit without opening truth. If you choose new candidate birth or a capability opening, specify the smallest bounded mechanism and why it attacks the residual.

## Constraints

Unless you explicitly justify `OPEN_CAPABILITY`, the next execution must remain:

- strict source geometry only;
- no truth, labels, CAD, drawings, nominal geometry, filename or part/family identity, A/B identity, expected count, or benchmark feedback during inference;
- deterministic CPU execution;
- approximately 200,000-point bounded detector input where applicable;
- peak RSS below 16 GiB and per-scan wall time below 300 seconds;
- compact outputs without source-point, source-vertex, source-face, membership-set, or support-bitset serialization;
- V52 retained and byte-frozen;
- V53, V54, and V55 terminal and unavailable for rescue;
- no runtime/main/production integration;
- frozen primary and replay order;
- immediate terminal failure on leakage, prefix mutation where a prefix is required, resource violation, schema overflow, non-finite output, or nondeterminism;
- native-resolution manual review of every changed case, all six exact V52 cases, and S011/S016/S024/S029 before retention.

The user explicitly authorizes source-free tests and synthetic data for rapid falsification, but the synthetic lane remains non-confirmatory and cannot justify enterprise promotion.

## Research grounding request

Ground the selected mechanism in relevant established research where possible: robust geometric normalization, line/axis representation, surface differential geometry, mesh-plane intersection and contour tracing, crease/ridge extraction, robust primitive fitting, graph optimization, or uncertainty calibration. Cite primary papers or authoritative methods and clearly distinguish those established ideas from Sherman-specific hypotheses. A citation is not evidence that the selected intervention will work here; the single bounded execution must remain falsifiable.

## What your response must specify

If `EXECUTE_ONE` or `OPEN_CAPABILITY`, provide:

1. exact intervention name, short name, precommit token, and terminal token;
2. exact insertion point and whether it appends, prunes, replaces, or creates a separate proposal path;
3. complete deterministic algorithm and every constant;
4. exact source inputs and exact candidate population, including a total pre-execution schema audit;
5. canonical line materialization, ordering, duplicate handling, and compact cache schema;
6. a direct transverse-localization certificate proving the reported axis is centered on local surface change rather than supported by offset folds;
7. scan-level and candidate-level abstention rules;
8. immediate failure gates;
9. a minimum positive retention gate materially above V52 and strict zero-regression rules for the six V52 exact cases;
10. mandatory native-resolution visual signatures and automatic visual failures, especially axis-normal section inspection;
11. leakage, determinism, latency, memory, and job-reliability audits;
12. causal explanation of why the mechanism attacks the current residual rather than retesting V53/V54 or rescuing V55;
13. research grounding with primary citations or named authoritative methods, separated from Sherman-specific assumptions.

Prefer one mechanism capable of improving several topology families. If the source-only observables are now insufficient, select the smallest capability opening and bind it to one testable execution rather than requesting a broad policy change.
